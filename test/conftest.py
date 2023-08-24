import pytest
from polyfactory.factories.pydantic_factory import ModelFactory

from stok.portfolio import PortfolioEntryModel


@pytest.fixture
def buy_action_factory():
    class BuyActionFactory(ModelFactory):
        __model__ = PortfolioEntryModel

    yield BuyActionFactory()
