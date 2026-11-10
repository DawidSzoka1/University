"""Po zalogowaniu koszyk gościa trafia do koszyka konta (z kumulacją ilości)."""

from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver

from apps.orders.cart import merge_guest_cart


@receiver(user_logged_in, dispatch_uid="orders_merge_guest_cart")
def merge_cart_after_login(sender, request, user, **kwargs):
    if request is not None:
        merge_guest_cart(request, user)
