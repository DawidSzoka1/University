from django.urls import path

from apps.orders import api, views

app_name = "orders"

urlpatterns = [
    # Koszyk
    path("koszyk/", views.cart_detail, name="cart"),
    path("koszyk/dodaj/<slug:slug>/", views.cart_add, name="cart_add"),
    path("koszyk/pozycja/<int:pk>/ilosc/", views.cart_update, name="cart_update"),
    path("koszyk/pozycja/<int:pk>/usun/", views.cart_remove, name="cart_remove"),
    # API koszyka (JWT)
    path("api/koszyk/", api.api_cart, name="api_cart"),
    path("api/koszyk/pozycje/", api.api_cart_add, name="api_cart_add"),
    path("api/koszyk/pozycje/<int:pk>/", api.api_cart_item, name="api_cart_item"),
    # Stałe adresy przed <number>/, żeby nie zostały wzięte za numer zamówienia
    path("zamowienie/", views.checkout, name="checkout"),
    path("obsluga/", views.manage_orders, name="manage"),
    path("kod/", views.redeem, name="redeem"),
    path("webhook/stripe/", views.stripe_webhook, name="stripe_webhook"),  # bez CSRF, podpis Stripe
    # Zamówienia ucznia i płatność
    path("", views.order_list, name="order_list"),
    path("<str:number>/", views.order_detail, name="order_detail"),
    path("<str:number>/zaplac/", views.order_pay, name="order_pay"),
    path("<str:number>/anuluj/", views.order_cancel, name="order_cancel"),
    path("<str:number>/platnosc/powrot/", views.payment_return, name="payment_return"),
    path("<str:number>/platnosc/przerwana/", views.payment_cancel, name="payment_cancel"),
    path("<str:number>/status/", views.order_status, name="order_status"),
]
