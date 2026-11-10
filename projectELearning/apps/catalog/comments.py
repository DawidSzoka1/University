"""Komentarze pod kursem: uprawnienia, drzewo odpowiedzi i powiadomienia.

Baza przechowuje pełne drzewo (``Comment.parent``) bez limitu głębokości. Widok wcina
odpowiedzi do ``MAX_DEPTH`` poziomów — głębsze trafiają płasko pod ostatni wcięty
komentarz z dopiskiem „w odpowiedzi do …” (jak w dużych serwisach dyskusyjnych). Dzięki
temu wątek mieści się na telefonie 320 px i da się go czytać czytnikiem ekranu.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from django.db.models import Count, Q
from django.urls import reverse
from django.utils import timezone

from apps.catalog.models import Comment, CommentVote, Course, Enrollment
from apps.catalog.permissions import course_reviewers, is_platform_admin, moderates_category
from apps.core.models import Notification
from apps.core.notifications import notify, notify_many

# Poziomy wcięcia: 0 (komentarz główny) … MAX_DEPTH. Głębsze odpowiedzi są spłaszczane.
MAX_DEPTH = 3


# --------------------------------------------------------------------- uprawnienia
def can_moderate_comment(user, comment: Comment) -> bool:
    return moderates_category(user, comment.course.category)


def can_see_comment(user, comment: Comment, *, is_moderator: bool) -> bool:
    if comment.is_removed:
        return False
    if comment.is_approved or is_moderator:
        return True
    return user.is_authenticated and comment.author_id == user.pk


def can_edit_comment(user, comment: Comment) -> bool:
    """Treść zmienia tylko autor (moderator może zatwierdzić albo usunąć, nie przepisać)."""
    return user.is_authenticated and not comment.is_removed and comment.author_id == user.pk


def can_delete_comment(user, comment: Comment) -> bool:
    if not user.is_authenticated or comment.is_removed:
        return False
    return comment.author_id == user.pk or can_moderate_comment(user, comment)


def can_reply_to(user, comment: Comment) -> bool:
    return user.is_authenticated and comment.is_public


def can_vote_on(user, comment: Comment) -> bool:
    return user.is_authenticated and comment.is_public and comment.author_id != user.pk


def can_rate_course(user, course: Course, *, exclude: Comment | None = None) -> bool:
    """Ocenę wystawia uczestnik kursu (nie autor), raz — w jednym komentarzu głównym."""
    if not user.is_authenticated or course.author_id == user.pk:
        return False
    if not Enrollment.objects.filter(user=user, course=course).exists():
        return False
    rated = Comment.objects.filter(
        course=course, author=user, parent__isnull=True, rating__isnull=False, is_removed=False
    )
    if exclude is not None:
        rated = rated.exclude(pk=exclude.pk)
    return not rated.exists()


def moderated_category_ids(user) -> list[int] | None:
    """Kategorie (z podkategoriami), które moderuje użytkownik. ``None`` = wszystkie (admin)."""
    if is_platform_admin(user):
        return None
    if not user.is_authenticated or getattr(user, "role", "") != "moderator":
        return []
    ids: set[int] = set()
    for category in user.moderated_categories.all():
        ids.update(category.get_descendants(include_self=True).values_list("id", flat=True))
    return sorted(ids)


def is_moderator_somewhere(user) -> bool:
    ids = moderated_category_ids(user)
    return ids is None or bool(ids)


def pending_comments_for(user):
    """Komentarze czekające na akceptację w kategoriach moderowanych przez użytkownika."""
    qs = Comment.objects.filter(is_approved=False, is_removed=False).select_related(
        "author", "course", "course__category", "parent__author"
    )
    ids = moderated_category_ids(user)
    if ids is not None:
        qs = qs.filter(course__category_id__in=ids)
    return qs.order_by("created_at")


def comment_url(comment: Comment) -> str:
    return reverse("catalog:course_detail", kwargs={"slug": comment.course.slug}) + f"#comment-{comment.pk}"


# ------------------------------------------------------------------ zmiany stanu
def approve_comment(comment: Comment, moderator) -> None:
    comment.is_approved = True
    comment.approved_by = moderator
    comment.approved_at = timezone.now()
    comment.save()
    if comment.author_id and comment.author_id != moderator.pk:
        notify(
            comment.author,
            Notification.Kind.COMMENT_APPROVED,
            "Twój komentarz został zaakceptowany",
            f"Komentarz pod kursem „{comment.course.title}” jest już widoczny.",
            url=comment_url(comment),
        )
    notify_reply(comment)


def notify_reply(comment: Comment) -> None:
    """Autor komentarza nadrzędnego dowiaduje się o odpowiedzi, gdy ta stanie się publiczna."""
    parent = comment.parent
    if parent is None or not comment.is_public or not parent.author_id or parent.author_id == comment.author_id:
        return
    notify(
        parent.author,
        Notification.Kind.COMMENT_REPLY,
        "Nowa odpowiedź na Twój komentarz",
        f"{_display_name(comment)} odpowiada pod kursem „{comment.course.title}”.",
        url=comment_url(comment),
    )


def notify_pending(comment: Comment) -> int:
    return notify_many(
        course_reviewers(comment.course),
        Notification.Kind.COMMENT_PENDING,
        "Komentarz czeka na akceptację",
        f"{_display_name(comment)} pod kursem „{comment.course.title}”.",
        url=reverse("catalog:moderation_comments"),
        exclude=comment.author,
    )


def remove_comment(comment: Comment) -> bool:
    """Usuwa komentarz. Z odpowiedziami — zostawia znacznik „usunięty”, żeby wątek się nie
    rozpadł (głosy i ocena znikają). Zwraca True, gdy zostawiono znacznik."""
    course = comment.course
    if comment.replies.exists():
        comment.votes.all().delete()  # sygnał przelicza reputację autora
        comment.is_removed = True
        comment.rating = None
        comment.save()
        return True
    comment.delete()
    course.recalculate_stats()
    return False


def toggle_vote(comment: Comment, user, value: int) -> int:
    """Ten sam przycisk drugi raz cofa głos, drugi przycisk zmienia go. Zwraca głos (0 = brak)."""
    vote = CommentVote.objects.filter(comment=comment, voter=user).first()
    if vote and vote.value == value:
        vote.delete()
        return 0
    if vote:
        vote.value = value
        vote.save()
    else:
        CommentVote.objects.create(comment=comment, voter=user, value=value)
    return value


def vote_counts(comment: Comment) -> tuple[int, int]:
    counts = comment.votes.aggregate(likes=Count("pk", filter=Q(value=1)), dislikes=Count("pk", filter=Q(value=-1)))
    return counts["likes"], counts["dislikes"]


# ------------------------------------------------------------------------ drzewo
@dataclass
class CommentNode:
    comment: Comment
    depth: int
    visible: bool  # False → wyświetlany jako „usunięty” / „ukryty” tylko dla zachowania wątku
    reply_to: str = ""  # imię autora, gdy odpowiedź spłaszczono pod innym komentarzem
    children: list[CommentNode] = field(default_factory=list)
    likes: int = 0
    dislikes: int = 0
    user_vote: int = 0
    can_edit: bool = False
    can_delete: bool = False
    can_reply: bool = False
    can_vote: bool = False
    can_approve: bool = False
    allow_rating: bool = False  # edycja w miejscu pokazuje wybór oceny kursu

    @property
    def reply_count(self) -> int:
        return sum(1 + child.reply_count for child in self.children)


def _display_name(comment: Comment) -> str:
    if not comment.author_id:
        return "konto usunięte"
    return comment.author.get_full_name() or comment.author.username


COMMENT_ORDERS = {
    "najnowsze": "Najnowsze",
    "najlepsze": "Najwyżej oceniane",
    "najstarsze": "Najstarsze",
}


def build_comment_tree(course: Course, user, order: str = "najnowsze") -> list[CommentNode]:
    """Wątki komentarzy kursu widoczne dla użytkownika. ``order`` dotyczy wątków głównych;
    odpowiedzi zawsze idą chronologicznie, jak rozmowa."""
    is_moderator = moderates_category(user, course.category)
    rate_ok = can_rate_course(user, course)
    comments = list(
        course.comments.select_related("author", "parent__author")
        .annotate(
            likes=Count("votes", filter=Q(votes__value=1)),
            dislikes=Count("votes", filter=Q(votes__value=-1)),
        )
        .order_by("created_at")
    )
    my_votes: dict[int, int] = {}
    if user.is_authenticated:
        my_votes = dict(
            CommentVote.objects.filter(comment__course=course, voter=user).values_list("comment_id", "value")
        )

    children: dict[int | None, list[Comment]] = defaultdict(list)
    for comment in comments:
        children[comment.parent_id].append(comment)

    def make(comment: Comment, depth: int, reply_to: str = "") -> CommentNode:
        visible = can_see_comment(user, comment, is_moderator=is_moderator)
        node = CommentNode(
            comment=comment,
            depth=depth,
            visible=visible,
            reply_to=reply_to,
            likes=comment.likes,
            dislikes=comment.dislikes,
            user_vote=my_votes.get(comment.pk, 0),
        )
        if visible:
            node.can_edit = can_edit_comment(user, comment)
            node.can_delete = node.can_edit or (is_moderator and not comment.is_removed)
            node.can_reply = can_reply_to(user, comment)
            node.can_vote = can_vote_on(user, comment)
            node.can_approve = is_moderator and not comment.is_approved
            node.allow_rating = node.can_edit and comment.parent_id is None and (
                comment.rating is not None or rate_ok
            )
        return node

    def flatten_into(parent_node: CommentNode, comment: Comment) -> None:
        # Wszystkie głębsze odpowiedzi jako rodzeństwo na poziomie MAX_DEPTH, chronologicznie
        for reply in children[comment.pk]:
            parent_node.children.append(make(reply, MAX_DEPTH, reply_to=_display_name(comment)))
            flatten_into(parent_node, reply)

    def build(comment: Comment, depth: int) -> CommentNode:
        node = make(comment, depth)
        if depth + 1 < MAX_DEPTH:
            node.children = [build(reply, depth + 1) for reply in children[comment.pk]]
        else:
            for reply in children[comment.pk]:
                node.children.append(make(reply, MAX_DEPTH))
                flatten_into(node, reply)
            node.children.sort(key=lambda child: child.comment.created_at)
        return node

    def prune(node: CommentNode) -> CommentNode | None:
        node.children = [kept for child in node.children if (kept := prune(child)) is not None]
        if node.visible or node.children:
            return node
        return None

    roots = [build(comment, 0) for comment in children[None]]  # chronologicznie
    if order == "najlepsze":
        roots.sort(key=lambda n: (n.likes - n.dislikes, n.likes, n.comment.created_at), reverse=True)
    elif order != "najstarsze":
        roots.reverse()
    return [kept for root in roots if (kept := prune(root)) is not None]


def public_comment_count(course: Course) -> int:
    return course.comments.filter(is_approved=True, is_removed=False).count()

