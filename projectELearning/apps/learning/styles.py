"""Style uczenia się: wybór stylu, wersja lekcji do pokazania i rekomendacja na podstawie wyników.

Rekomendacja
------------
Każde ocenione podejście do testu ma zapisany styl, w którym uczeń przerabiał lekcje kursu
(``style_for_attempt``). Dla każdego stylu liczymy średni wynik procentowy podejść i ich liczbę.
Żeby jedno szczęśliwe podejście nie przesądzało o wyniku, średnią „ściągamy” do ogólnej
średniej ucznia (średnia ważona z wagą ``PRIOR_WEIGHT``):

    wynik_stylu = (n · średnia_stylu + K · średnia_ogólna) / (n + K)

Styl jest polecany, gdy ma co najmniej ``MIN_ATTEMPTS`` podejść i jego wynik przewyższa
wynik obecnego stylu o co najmniej ``MIN_ADVANTAGE`` punktów procentowych. Przy remisie
decyduje odsetek zaliczonych testów, potem liczba ukończonych lekcji w tym stylu.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from django.db.models import QuerySet
from django.utils import timezone

from apps.learning.models import Attempt, LearningStyle, Lesson, LessonProgress, LessonVariant

BASE_SLUG = "podstawowa"  # wersja podstawowa lekcji w parametrze ?wersja=
MIN_ATTEMPTS = 2
PRIOR_WEIGHT = 2
MIN_ADVANTAGE = 5


def active_styles() -> QuerySet[LearningStyle]:
    return LearningStyle.objects.filter(is_active=True)


def user_style(user) -> LearningStyle | None:
    """Styl wybrany przez ucznia, o ile nadal jest aktywny."""
    if not getattr(user, "is_authenticated", False):
        return None
    style = getattr(user, "learning_style", None)
    return style if style is not None and style.is_active else None


def set_user_style(user, style: LearningStyle | None) -> None:
    user.learning_style = style
    user.learning_style_updated_at = timezone.now()
    user.save(update_fields=["learning_style", "learning_style_updated_at"])


# ---------------------------------------------------------------- wersja lekcji
@dataclass
class LessonVersion:
    """Co pokazać w lekcji: wariant dla stylu albo treść podstawowa (variant = None)."""

    lesson: Lesson
    variant: LessonVariant | None = None
    available: list[LessonVariant] = field(default_factory=list)
    own_style: LearningStyle | None = None
    requested: bool = False  # uczeń sam wybrał wersję parametrem ?wersja=

    @property
    def style(self) -> LearningStyle | None:
        return self.variant.style if self.variant else None

    @property
    def content_html(self) -> str:
        return self.variant.content_html if self.variant else self.lesson.content_html

    @property
    def video_url(self) -> str:
        if self.variant and self.variant.video_url:
            return self.variant.video_url
        return self.lesson.video_url

    @property
    def uses_variant_video(self) -> bool:
        return bool(self.variant and self.variant.video_url)

    @property
    def has_variants(self) -> bool:
        return bool(self.available)

    @property
    def own_variant_missing(self) -> bool:
        """Uczeń ma styl, lekcja ma wersje, ale nie dla jego stylu."""
        return bool(self.available and self.own_style and all(v.style_id != self.own_style.pk for v in self.available))


def lesson_variants(lesson: Lesson) -> list[LessonVariant]:
    if lesson.same_for_all_styles:
        return []
    return list(lesson.variants.filter(style__is_active=True).select_related("style"))


def resolve_version(lesson: Lesson, user, requested: str | None = None) -> LessonVersion:
    """Wersja dla stylu ucznia; ``requested`` (slug lub ``podstawowa``) pozwala obejrzeć inną."""
    own = user_style(user)
    available = lesson_variants(lesson)
    version = LessonVersion(lesson=lesson, available=available, own_style=own)
    by_slug = {variant.style.slug: variant for variant in available}
    if requested:
        if requested == BASE_SLUG:
            version.requested = True
            return version
        if requested in by_slug:
            version.variant, version.requested = by_slug[requested], True
            return version
    if own is not None:
        version.variant = next((v for v in available if v.style_id == own.pk), None)
    return version


def variant_counts(lessons) -> dict[int, int]:
    """Liczba aktywnych wersji dla każdej lekcji (do programu kursu w kreatorze)."""
    counts = Counter(
        LessonVariant.objects.filter(lesson__in=lessons, style__is_active=True).values_list("lesson_id", flat=True)
    )
    return dict(counts)


# ------------------------------------------------------------ analiza wyników
def style_for_attempt(user, quiz) -> LearningStyle | None:
    """Styl, w którym uczeń przerabiał lekcje tego kursu; brak danych → jego obecny styl."""
    styles = (
        LessonProgress.objects.filter(user=user, lesson__section__course=quiz.course, style__isnull=False)
        .values_list("style_id", flat=True)
    )
    common = Counter(styles).most_common(1)
    if common:
        return LearningStyle.objects.filter(pk=common[0][0]).first()
    return user_style(user)


@dataclass
class StyleStats:
    style: LearningStyle
    attempts: int = 0
    passed: int = 0
    percent_sum: int = 0
    lessons: int = 0
    score: float | None = None  # wynik ważony (patrz opis modułu)

    @property
    def average(self) -> int | None:
        return round(self.percent_sum / self.attempts) if self.attempts else None

    @property
    def pass_rate(self) -> int | None:
        return round(100 * self.passed / self.attempts) if self.attempts else None

    @property
    def score_display(self) -> int | None:
        return round(self.score) if self.score is not None else None


@dataclass
class Recommendation:
    rows: list[StyleStats]
    current: LearningStyle | None
    suggested: LearningStyle | None = None
    kind: str = "no_data"  # no_data | keep | switch | choose | explore
    graded_attempts: int = 0
    overall_average: int | None = None
    explore: LearningStyle | None = None

    @property
    def differs(self) -> bool:
        return self.kind in {"switch", "choose"} and self.suggested is not None


def analyse(user) -> Recommendation:
    styles = list(active_styles())
    stats = {style.pk: StyleStats(style=style) for style in styles}
    attempts = Attempt.objects.filter(user=user, status=Attempt.Status.GRADED, max_score__gt=0)
    percents = []
    for attempt in attempts.only("score", "max_score", "passed", "learning_style_id"):
        percents.append(attempt.percent)
        row = stats.get(attempt.learning_style_id)
        if row is None:
            continue
        row.attempts += 1
        row.percent_sum += attempt.percent
        row.passed += int(attempt.passed)
    for style_id, count in Counter(
        LessonProgress.objects.filter(user=user, style__isnull=False).values_list("style_id", flat=True)
    ).items():
        if style_id in stats:
            stats[style_id].lessons = count

    current = user_style(user)
    result = Recommendation(rows=list(stats.values()), current=current, graded_attempts=len(percents))
    if not percents:
        result.explore = _style_to_try(result.rows, current)
        return result
    overall = sum(percents) / len(percents)
    result.overall_average = round(overall)
    for row in result.rows:
        if row.attempts:
            row.score = (row.attempts * (row.percent_sum / row.attempts) + PRIOR_WEIGHT * overall) / (row.attempts + PRIOR_WEIGHT)

    candidates = [row for row in result.rows if row.attempts >= MIN_ATTEMPTS]
    if not candidates:
        result.explore = _style_to_try(result.rows, current)
        return result
    best = max(candidates, key=lambda r: (r.score, r.pass_rate or 0, r.lessons))
    current_row = stats.get(current.pk) if current else None
    if current is None:
        result.kind, result.suggested = "choose", best.style
    elif best.style.pk == current.pk:
        result.kind, result.suggested = "keep", current
    elif current_row is None or current_row.score is None or best.score - current_row.score >= MIN_ADVANTAGE:
        result.kind, result.suggested = "switch", best.style
    else:
        result.kind, result.suggested = "keep", current
    result.explore = _style_to_try(result.rows, current)
    return result


def _style_to_try(rows: list[StyleStats], current: LearningStyle | None) -> LearningStyle | None:
    """Styl jeszcze niesprawdzony w testach — propozycja, żeby zebrać porównanie."""
    for row in rows:
        if row.attempts == 0 and (current is None or row.style.pk != current.pk):
            return row.style
    return None


# ---------------------------------------------------------------- ankieta stylu
def survey_questions():
    """Aktywne pytania z odpowiedziami tylko dla aktywnych stylów (pytania bez odpowiedzi pomijane)."""
    from django.db.models import Prefetch

    from apps.learning.models import SurveyOption, SurveyQuestion

    options = SurveyOption.objects.filter(style__is_active=True).select_related("style")
    questions = SurveyQuestion.objects.filter(is_active=True).prefetch_related(Prefetch("options", queryset=options))
    return [question for question in questions if question.options.all()]


def latest_survey(user):
    from apps.learning.models import SurveyResponse

    if not getattr(user, "is_authenticated", False):
        return None
    return SurveyResponse.objects.filter(user=user).select_related("top_style").first()


def submit_survey(user, picked: dict):
    """Zapisuje odpowiedzi (pytanie → odpowiedź) i wyznacza dominujący styl.

    Wynik stylu = liczba odpowiedzi wskazujących styl / liczba pytań. Remis rozstrzyga
    kolejność stylów (pole ``order``), żeby ten sam zestaw odpowiedzi dawał zawsze ten sam styl.
    """
    from apps.learning.models import SurveyAnswer, SurveyResponse

    counts = Counter(option.style_id for option in picked.values())
    styles = list(active_styles())
    top = max(styles, key=lambda s: (counts.get(s.pk, 0), -styles.index(s)), default=None)
    response = SurveyResponse.objects.create(user=user, top_style=top if top and counts.get(top.pk) else None)
    SurveyAnswer.objects.bulk_create(
        SurveyAnswer(response=response, question=question, option=option) for question, option in picked.items()
    )
    return response


@dataclass
class SurveyShare:
    style: LearningStyle
    count: int
    percent: int


def survey_profile(response) -> list[SurveyShare]:
    if response is None:
        return []
    counts = Counter(response.answers.values_list("option__style_id", flat=True))
    total = sum(counts.values()) or 1
    return [SurveyShare(style, counts.get(style.pk, 0), round(100 * counts.get(style.pk, 0) / total)) for style in active_styles()]


# ------------------------------------------------------------ profil i polecanie kursów
CHOSEN_WEIGHT = 1.0      # styl wybrany przez ucznia
SURVEY_WEIGHT = 1.0      # ankieta (rozkładana proporcjonalnie na style)
RESULTS_WEIGHT = 0.5     # styl, w którym uczniowi najlepiej idą testy
VARIANT_SHARE = 0.7      # udział lekcji z wersją dla stylu
DECLARED_SHARE = 0.3     # autor zaznaczył kurs jako szczególnie dla stylu


@dataclass
class LearnerProfile:
    weights: dict[int, float]          # style_id → udział (suma 1)
    sources: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.weights


def learner_profile(user) -> LearnerProfile:
    """Profil stylów ucznia: wybrany styl + wynik ankiety + styl najlepszych wyników testów."""
    raw: Counter = Counter()
    sources = []
    if not getattr(user, "is_authenticated", False):
        return LearnerProfile({}, [])
    chosen = user_style(user)
    if chosen:
        raw[chosen.pk] += CHOSEN_WEIGHT
        sources.append(f"wybrany styl „{chosen.name}”")
    survey = latest_survey(user)
    shares = [s for s in survey_profile(survey) if s.count]
    if shares:
        total = sum(s.count for s in shares)
        for share in shares:
            raw[share.style.pk] += SURVEY_WEIGHT * share.count / total
        sources.append("wynik ankiety")
    analysis = analyse(user)
    if analysis.suggested is not None and analysis.kind in {"keep", "switch", "choose"}:
        raw[analysis.suggested.pk] += RESULTS_WEIGHT
        sources.append("wyniki testów")
    total = sum(raw.values())
    return LearnerProfile({k: v / total for k, v in raw.items()} if total else {}, sources)


@dataclass
class CourseFit:
    course: object
    score: float
    reason: str
    style: LearningStyle | None = None


def course_style_fit(courses) -> dict[int, dict[int, tuple[float, int, int, bool]]]:
    """course_id → style_id → (dopasowanie 0–1, lekcje z wersją, wszystkie lekcje, zaznaczony przez autora)."""
    from django.db.models import Count

    from apps.catalog.models import Course

    ids = [c.pk for c in courses]
    totals = dict(
        Lesson.objects.filter(section__course_id__in=ids).values_list("section__course_id").annotate(n=Count("pk"))
    )
    with_variant = Counter()
    for course_id, style_id in (
        LessonVariant.objects.filter(
            lesson__section__course_id__in=ids, lesson__same_for_all_styles=False, style__is_active=True
        ).values_list("lesson__section__course_id", "style_id")
    ):
        with_variant[(course_id, style_id)] += 1
    declared = set(
        Course.recommended_styles.through.objects.filter(course_id__in=ids, learningstyle__is_active=True)
        .values_list("course_id", "learningstyle_id")
    )
    result: dict[int, dict[int, tuple[float, int, int, bool]]] = {}
    for style in active_styles():
        for course_id in ids:
            total = totals.get(course_id, 0)
            count = with_variant.get((course_id, style.pk), 0)
            is_declared = (course_id, style.pk) in declared
            fit = VARIANT_SHARE * (count / total if total else 0) + DECLARED_SHARE * is_declared
            if fit:
                result.setdefault(course_id, {})[style.pk] = (fit, count, total, is_declared)
    return result


def rank_courses(user, courses, profile: LearnerProfile | None = None) -> list[CourseFit]:
    """Dopasowanie = Σ udział_stylu_ucznia · dopasowanie_kursu_do_stylu. Malejąco; zera pomijane."""
    profile = profile or learner_profile(user)
    courses = list(courses)
    if profile.is_empty or not courses:
        return []
    fits = course_style_fit(courses)
    names = {style.pk: style for style in active_styles()}
    ranked = []
    for course in courses:
        per_style = fits.get(course.pk, {})
        contributions = {sid: weight * per_style[sid][0] for sid, weight in profile.weights.items() if sid in per_style}
        score = sum(contributions.values())
        if score <= 0:
            continue
        best = max(contributions, key=contributions.get)
        _, count, total, is_declared = per_style[best]
        style = names[best]
        if count:
            reason = f"{count} z {total} lekcji ma wersję dla stylu „{style.name}”"
            if is_declared:
                reason += "; autor poleca kurs dla tego stylu"
        else:
            reason = f"Autor poleca kurs dla stylu „{style.name}”"
        course.fit_reason = reason
        ranked.append(CourseFit(course=course, score=score, reason=reason, style=style))
    ranked.sort(key=lambda f: (-f.score, -float(f.course.average_rating or 0), -f.course.students_count))
    return ranked


def recommend_courses(user, limit: int = 3) -> list[CourseFit]:
    """Polecane kursy, na które uczeń jeszcze się nie zapisał."""
    from apps.catalog.models import Course

    if not getattr(user, "is_authenticated", False):
        return []
    courses = Course.objects.published().exclude(enrollments__user=user).select_related("category", "author")
    return rank_courses(user, courses)[:limit]
