from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline

from apps.orders.models import AccessCode, Cart, CartItem, Order, OrderItem, OrderStatusChange


class OrderItemInline(TabularInline):
    model = OrderItem
    extra = 0
    raw_id_fields = ("course",)


class OrderStatusChangeInline(TabularInline):
    model = OrderStatusChange
    extra = 0
    fields = ("status", "note", "changed_by", "created_at")
    readonly_fields = fields

    def has_add_permission(self, request, obj=None) -> bool:
        return False


@admin.register(Order)
class OrderAdmin(ModelAdmin):
    list_display = ("number", "user", "status", "total", "payment_provider", "created_at", "paid_at")
    list_filter = ("status", "payment_provider", "created_at")
    search_fields = ("number", "user__username", "email", "last_name", "payment_reference")
    raw_id_fields = ("user",)
    readonly_fields = ("payment_reference", "payment_intent", "paid_at", "fulfilled_at", "created_at", "updated_at")
    inlines = [OrderItemInline, OrderStatusChangeInline]


@admin.register(AccessCode)
class AccessCodeAdmin(ModelAdmin):
    list_display = ("code", "order_item", "redeemed_by", "redeemed_at")
    search_fields = ("code", "order_item__order__number", "redeemed_by__username")
    raw_id_fields = ("order_item", "redeemed_by")


class CartItemInline(TabularInline):
    model = CartItem
    extra = 0
    raw_id_fields = ("course",)


@admin.register(Cart)
class CartAdmin(ModelAdmin):
    list_display = ("pk", "user", "updated_at")
    raw_id_fields = ("user",)
    inlines = [CartItemInline]
