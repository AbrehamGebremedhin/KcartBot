from app.db.models import CompetitorPrice
from tortoise.exceptions import DoesNotExist

class CompetitorPriceRepository:
    @staticmethod
    async def create_competitor_price(**kwargs):
        return await CompetitorPrice.create(**kwargs)

    @staticmethod
    async def get_competitor_price_by_id(id):
        try:
            return await CompetitorPrice.get(id=id).fetch_related('product')
        except DoesNotExist:
            return None

    @staticmethod
    async def update_competitor_price(id, **kwargs):
        competitor_price = await CompetitorPriceRepository.get_competitor_price_by_id(id)
        if competitor_price:
            for key, value in kwargs.items():
                setattr(competitor_price, key, value)
            await competitor_price.save()
        return competitor_price

    @staticmethod
    async def delete_competitor_price(id):
        competitor_price = await CompetitorPriceRepository.get_competitor_price_by_id(id)
        if competitor_price:
            await competitor_price.delete()
            return True
        return False

    @staticmethod
    async def list_competitor_prices(filters=None):
        query = CompetitorPrice.all().prefetch_related('product')
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
