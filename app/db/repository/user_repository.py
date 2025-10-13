from app.db.models import User
from tortoise.exceptions import DoesNotExist

class UserRepository:
    @staticmethod
    async def create_user(**kwargs):
        return await User.create(**kwargs)

    @staticmethod
    async def get_user_by_name(name: str):
        if not name:
            return None
        return await User.filter(name__iexact=name).first()

    @staticmethod
    async def get_user_by_id(user_id):
        try:
            return await User.get(user_id=user_id)
        except DoesNotExist:
            return None

    @staticmethod
    async def update_user(user_id, **kwargs):
        user = await UserRepository.get_user_by_id(user_id)
        if user:
            for key, value in kwargs.items():
                setattr(user, key, value)
            await user.save()
        return user

    @staticmethod
    async def delete_user(user_id):
        user = await UserRepository.get_user_by_id(user_id)
        if user:
            await user.delete()
            return True
        return False

    @staticmethod
    async def list_users(filters=None):
        query = User.all()
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    query = query.filter(**{f"{key}__{value['lookup']}": value['value']})
                else:
                    query = query.filter(**{key: value})
        return await query
