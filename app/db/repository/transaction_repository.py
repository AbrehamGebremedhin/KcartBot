from app.db.models import Transaction
from tortoise.exceptions import DoesNotExist

class TransactionRepository:
    @staticmethod
    async def create_transaction(**kwargs):
        return await Transaction.create(**kwargs)

    @staticmethod
    async def get_transaction_by_id(order_id):
        try:
            transaction = await Transaction.get(order_id=order_id)
            await transaction.fetch_related('user')
            return transaction
        except DoesNotExist:
            return None

    @staticmethod
    async def update_transaction(order_id, **kwargs):
        transaction = await TransactionRepository.get_transaction_by_id(order_id)
        if transaction:
            for key, value in kwargs.items():
                setattr(transaction, key, value)
            await transaction.save()
        return transaction

    @staticmethod
    async def delete_transaction(order_id):
        transaction = await TransactionRepository.get_transaction_by_id(order_id)
        if transaction:
            await transaction.delete()
            return True
        return False

    @staticmethod
    async def list_transactions(filters=None):
        query = Transaction.all().prefetch_related('user')
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
