"""Sygnały katalogu: przeliczanie reputacji po każdej zmianie głosu."""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from apps.catalog.models import Comment, CommentVote


def _recalculate_comment_author(comment_id: int) -> None:
    author_id = Comment.objects.filter(pk=comment_id).values_list("author_id", flat=True).first()
    if author_id is None:
        return
    author = get_user_model().objects.filter(pk=author_id).first()
    if author is not None:
        author.recalculate_reputation()


@receiver(post_save, sender=CommentVote, dispatch_uid="catalog_vote_saved_reputation")
@receiver(post_delete, sender=CommentVote, dispatch_uid="catalog_vote_deleted_reputation")
def update_reputation_after_vote(sender, instance: CommentVote, **kwargs) -> None:
    """Nowy, zmieniony (+1 ↔ −1) albo usunięty głos zmienia punkty i rangę autora."""
    _recalculate_comment_author(instance.comment_id)
