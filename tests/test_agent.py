"""Tests for the KCartBot Agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.agent import Agent


class TestAgent:
    """Test cases for the KCartBot Agent."""

    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service."""
        return MagicMock()

    @pytest.fixture
    def mock_intent_classifier(self):
        """Mock intent classifier tool."""
        mock_tool = AsyncMock()
        mock_tool.run.return_value = {
            "intent": "intent.customer.register",
            "flow": "customer",
            "filled_slots": {"customer_name": "John Doe"},
            "missing_slots": ["phone_number"],
            "suggested_tools": []
        }
        return mock_tool

    @pytest.fixture
    def mock_database_tool(self):
        """Mock database access tool."""
        mock_tool = AsyncMock()
        mock_tool.run.return_value = {"user_id": 1, "name": "John Doe"}
        return mock_tool

    @pytest.fixture
    def mock_vector_search(self):
        """Mock vector search tool."""
        mock_tool = AsyncMock()
        mock_tool.run.return_value = {
            "results": [{"text": "Store in cool, dry place"}],
            "error": None
        }
        return mock_tool

    @pytest.fixture
    def mock_date_resolver(self):
        """Mock date resolver tool."""
        mock_tool = AsyncMock()
        mock_tool.run.return_value = MagicMock()  # Mock date object
        return mock_tool

    @pytest.fixture
    def mock_image_generator(self):
        """Mock image generator tool."""
        mock_tool = AsyncMock()
        mock_tool.run.return_value = "image_url.jpg"
        return mock_tool

    @pytest.fixture
    def agent(self, mock_llm_service, mock_intent_classifier, mock_database_tool,
              mock_vector_search, mock_date_resolver, mock_image_generator):
        """Create agent with mocked dependencies."""
        with patch('app.agents.agent.LLMService', return_value=mock_llm_service), \
             patch('app.agents.agent.IntentClassifierTool', return_value=mock_intent_classifier), \
             patch('app.agents.agent.DatabaseAccessTool', return_value=mock_database_tool), \
             patch('app.agents.agent.VectorSearchTool', return_value=mock_vector_search), \
             patch('app.agents.agent.DateResolverTool', return_value=mock_date_resolver), \
             patch('app.agents.agent.ImageGeneratorTool', return_value=mock_image_generator):
            return Agent()

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.llm_service is not None
        assert agent.intent_classifier is not None
        assert agent.database_tool is not None
        assert agent.vector_search is not None
        assert agent.date_resolver is not None
        assert agent.image_generator is not None
        assert "intent_classifier" in agent.tools
        assert "database_access" in agent.tools

    @pytest.mark.asyncio
    async def test_process_message_customer_registration(self, agent, mock_intent_classifier, mock_database_tool):
        """Test processing a customer registration message."""
        # Setup mocks
        mock_intent_classifier.run.return_value = {
            "intent": "intent.customer.register",
            "flow": "customer",
            "filled_slots": {"customer_name": "John Doe", "phone_number": "0912345678", "default_location": "Addis Ababa"},
            "missing_slots": [],
            "suggested_tools": []
        }
        mock_database_tool.run.return_value = {"user_id": 1, "name": "John Doe"}

        # Process message
        result = await agent.process_message("I want to register as a customer")

        # Assertions
        assert "response" in result
        assert "session_context" in result
        assert "intent_info" in result
        assert result["intent_info"]["intent"] == "intent.customer.register"
        assert result["intent_info"]["flow"] == "customer"
        assert result["session_context"]["user_id"] == 1
        assert "Welcome John Doe!" in result["response"]

    @pytest.mark.asyncio
    async def test_process_message_unknown_intent(self, agent, mock_intent_classifier):
        """Test processing a message with unknown intent."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.unknown",
            "flow": "unknown",
            "filled_slots": {},
            "missing_slots": [],
            "suggested_tools": []
        }

        result = await agent.process_message("Hello there")

        assert "response" in result
        assert "Are you a customer looking to place an order" in result["response"]

    @pytest.mark.asyncio
    async def test_process_message_with_chat_history(self, agent, mock_intent_classifier):
        """Test processing message with existing chat history."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.unknown",
            "flow": "unknown",
            "filled_slots": {},
            "missing_slots": [],
            "suggested_tools": []
        }

        session_context = {
            "chat_history": [
                {"role": "user", "content": "I want to buy tomatoes"},
                {"role": "assistant", "content": "Great! What quantity would you like?"}
            ]
        }

        result = await agent.process_message("Yes, 5 kg please", session_context)

        assert "response" in result
        assert "Could you please provide more details" in result["response"]

    @pytest.mark.asyncio
    async def test_customer_supplier_availability(self, agent, mock_intent_classifier, mock_database_tool):
        """Test customer checking products available from a supplier."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.customer.check_availability",
            "flow": "customer",
            "filled_slots": {"product_name": "Edom"},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls: first find_product_by_any_name (returns None), then get_user_by_name, then list_supplier_products
        mock_database_tool.run.side_effect = [
            None,  # Product not found
            {"user_id": 71, "name": "Edom", "role": "supplier"},  # Supplier found
            [  # Supplier products
                {
                    "product": {"product_name_en": "Clarified butter / Ghee"},
                    "quantity_available": 50.0,
                    "unit_price_etb": 150.0,
                    "unit": "kg"
                },
                {
                    "product": {"product_name_en": "Mango"},
                    "quantity_available": 100.0,
                    "unit_price_etb": 75.0,
                    "unit": "kg"
                }
            ]
        ]

        result = await agent.process_message("what can i order from Edom?")

        assert "response" in result
        assert "Products available from Edom:" in result["response"]
        assert "Clarified butter / Ghee" in result["response"]
        assert "Mango" in result["response"]
        assert "150.0 ETB per kg" in result["response"]
        assert "75.0 ETB per kg" in result["response"]

    @pytest.mark.asyncio
    async def test_customer_storage_advice(self, agent, mock_intent_classifier, mock_vector_search):
        """Test customer asking for storage advice."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.customer.storage_advice",
            "flow": "customer",
            "filled_slots": {"product_name": "apples"},
            "missing_slots": [],
            "suggested_tools": []
        }

        result = await agent.process_message("How should I store apples?")

        assert "response" in result
        assert "apples" in result["response"]
        assert "cool, dry place" in result["response"]

    @pytest.mark.asyncio
    async def test_supplier_registration(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier registration."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.register",
            "flow": "supplier",
            "filled_slots": {"supplier_name": "Green Farms", "phone_number": "0912345678"},
            "missing_slots": [],
            "suggested_tools": []
        }

        mock_database_tool.run.return_value = {"user_id": 2, "name": "Green Farms"}

        result = await agent.process_message("I want to register as a supplier")

        assert "response" in result
        assert "Welcome Green Farms!" in result["response"]
        assert result["session_context"]["user_id"] == 2

    @pytest.mark.asyncio
    async def test_supplier_add_product(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier adding a product."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.add_product",
            "flow": "supplier",
            "filled_slots": {"product_name": "carrots"},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls for product creation
        mock_database_tool.run.side_effect = [
            None,  # Product not found by name
            {"product_id": 1, "product_name_en": "carrots"},  # Created product
            []  # No existing supplier products
        ]

        session_context = {"user_id": 2}
        result = await agent.process_message("I want to add carrots", session_context)

        assert "response" in result
        assert "carrots" in result["response"]
        assert "quantity you have available" in result["response"]

    @pytest.mark.asyncio
    async def test_supplier_check_stock(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier checking their stock."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.check_stock",
            "flow": "supplier",
            "filled_slots": {},
            "missing_slots": [],
            "suggested_tools": []
        }

        mock_database_tool.run.return_value = [
            {
                "product": {"product_name_en": "tomatoes"},
                "quantity_available": 50,
                "unit": "kg",
                "unit_price_etb": 25.0,
                "expiry_date": "2025-10-25",
                "available_delivery_days": "Monday-Friday",
                "status": "active"
            },
            {
                "product": {"product_name_en": "carrots"},
                "quantity_available": 30,
                "unit": "kg",
                "unit_price_etb": 15.0,
                "expiry_date": None,
                "available_delivery_days": "Weekends",
                "status": "active"
            }
        ]

        session_context = {"user_id": 2}
        result = await agent.process_message("What's my current stock?", session_context)

        assert "response" in result
        assert "📦 **Your Current Inventory:**" in result["response"]
        assert "✅ **tomatoes**" in result["response"]
        assert "Quantity: 50 kg" in result["response"]
        assert "Price: 25.0 ETB/kg" in result["response"]
        assert "Delivery: Monday-Friday" in result["response"]
        assert "✅ **carrots**" in result["response"]
        assert "Quantity: 30 kg" in result["response"]
        assert "Price: 15.0 ETB/kg" in result["response"]

    @pytest.mark.asyncio
    async def test_error_handling(self, agent, mock_intent_classifier):
        """Test error handling in message processing."""
        # Make intent classifier raise an exception
        mock_intent_classifier.run.side_effect = Exception("Test error")

        result = await agent.process_message("Hello")

        assert "response" in result
        assert "error" in result
        assert "encountered an error" in result["response"]

    @pytest.mark.asyncio
    async def test_missing_slots_handling(self, agent, mock_intent_classifier):
        """Test handling of missing slots in customer registration."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.customer.register",
            "flow": "customer",
            "filled_slots": {"customer_name": "John Doe"},
            "missing_slots": ["phone_number"],
            "suggested_tools": []
        }

        result = await agent.process_message("I want to register")

        assert "response" in result
        assert "phone number" in result["response"]

    @pytest.mark.asyncio
    async def test_chat_history_management(self, agent, mock_intent_classifier):
        """Test that chat history is properly managed."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.unknown",
            "flow": "unknown",
            "filled_slots": {},
            "missing_slots": [],
            "suggested_tools": []
        }

        # First message
        result1 = await agent.process_message("Hello")

        # Second message with existing context
        result2 = await agent.process_message("I need help", result1["session_context"])

        assert len(result2["session_context"]["chat_history"]) == 4  # 2 from first + 2 from second
        assert result2["session_context"]["chat_history"][-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_process_message_with_greeting(self, agent, mock_intent_classifier):
        """Test processing a greeting message."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.unknown",
            "flow": "unknown",
            "filled_slots": {},
            "missing_slots": [],
            "suggested_tools": []
        }

        result = await agent.process_message("hello")

        assert "response" in result
        assert "Hello! Welcome to KCartBot" in result["response"]
        assert "customer looking to place an order" in result["response"]
        assert "supplier managing inventory" in result["response"]

    @pytest.mark.asyncio
    async def test_customer_place_order_flow(self, agent, mock_intent_classifier, mock_database_tool):
        """Test the customer order placement flow."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.customer.place_order",
            "flow": "customer",
            "filled_slots": {
                "order_items": [
                    {"product_name": "tomatoes", "quantity": 5}
                ]
            },
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls for order placement
        # 1. find_product_by_any_name, 2. list_supplier_products, 3. get_user_by_id (user instance),
        # 4. create_transaction, 5. get_product_by_id (product instance), 6. get_user_by_id (supplier instance),
        # 7. create_order_item
        mock_database_tool.run.side_effect = [
            {"product_id": 1, "product_name_en": "tomatoes"},  # find_product_by_any_name
            [{"product": {"product_id": 1}, "supplier": {"user_id": 2}, "unit_price_etb": 25.0}],  # list_supplier_products
            MagicMock(),  # User model instance
            {"order_id": 100},  # Transaction creation
            MagicMock(),  # Product model instance
            MagicMock(),  # Supplier model instance
            None  # Order item creation
        ]

        session_context = {"user_id": 1}
        result = await agent.process_message("I want to order 5kg tomatoes", session_context)

        assert "response" in result
        assert "Order placed successfully" in result["response"]
        assert "125.0 ETB" in result["response"]  # 5 * 25

    @pytest.mark.asyncio
    async def test_supplier_set_quantity_with_pricing_suggestions(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier setting quantity with pricing suggestions."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.set_quantity",
            "flow": "supplier",
            "filled_slots": {"product_name": "tomatoes", "quantity": 50},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls: first find_product_by_any_name, then list_competitor_prices
        mock_database_tool.run.side_effect = [
            {"product_id": 1, "product_name_en": "tomatoes"},  # Product found
            [  # Competitor prices
                {"price_etb_per_kg": 20.0},
                {"price_etb_per_kg": 25.0},
                {"price_etb_per_kg": 30.0}
            ]
        ]

        session_context = {"user_id": 2, "pending_product": {"product_name": "tomatoes"}}
        result = await agent.process_message("50 kg", session_context)

        assert "response" in result
        assert "I'll add 50 kg of tomatoes" in result["response"]
        assert "Market Insights for tomatoes" in result["response"]
        assert "Average market price: 25.0 ETB/kg" in result["response"]
        assert "Suggested competitive range" in result["response"]
        assert "What's the price per kg" in result["response"]

    @pytest.mark.asyncio
    async def test_supplier_set_price_asks_for_expiry(self, agent, mock_intent_classifier):
        """Test supplier setting price asks for expiry date."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.set_price",
            "flow": "supplier",
            "filled_slots": {"unit_price": 25},
            "missing_slots": [],
            "suggested_tools": []
        }

        session_context = {
            "user_id": 2,
            "pending_product": {
                "product_name": "tomatoes",
                "quantity": 50
            }
        }
        result = await agent.process_message("25", session_context)

        assert "response" in result
        assert "Great! I'll set the price at 25 ETB per kg" in result["response"]
        assert "When does this tomatoes expire" in result["response"]
        assert result["session_context"]["pending_product"]["unit_price"] == 25

    @pytest.mark.asyncio
    async def test_supplier_set_expiry_date_asks_for_delivery(self, agent, mock_intent_classifier, mock_date_resolver):
        """Test supplier setting expiry date asks for delivery dates."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.set_expiry_date",
            "flow": "supplier",
            "filled_slots": {"expiry_date": "next week"},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock date resolver
        mock_date_resolver.run.return_value = MagicMock()
        mock_date_resolver.run.return_value.isoformat.return_value = "2025-10-23"
        mock_date_resolver.run.return_value.strftime.return_value = "October 23, 2025"

        session_context = {
            "user_id": 2,
            "pending_product": {
                "product_name": "tomatoes",
                "quantity": 50,
                "unit_price": 25
            }
        }
        result = await agent.process_message("next week", session_context)

        assert "response" in result
        assert "Expiry date set to October 23, 2025" in result["response"]
        assert "What days can you deliver tomatoes" in result["response"]
        assert result["session_context"]["pending_product"]["expiry_date"] == "2025-10-23"

    @pytest.mark.asyncio
    async def test_supplier_set_delivery_dates_creates_product(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier setting delivery dates creates the product."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.set_delivery_dates",
            "flow": "supplier",
            "filled_slots": {"delivery_dates": "Monday to Friday"},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls for product creation
        mock_database_tool.run.side_effect = [
            {"product_id": 1, "product_name_en": "tomatoes"},  # find_product_by_any_name
            MagicMock(),  # User instance
            MagicMock(),  # Product instance
            [],  # No existing supplier products
            None  # Product creation success
        ]

        session_context = {
            "user_id": 2,
            "pending_product": {
                "product_name": "tomatoes",
                "quantity": 50,
                "unit_price": 25,
                "expiry_date": "2025-10-23"
            }
        }
        result = await agent.process_message("Monday to Friday", session_context)

        assert "response" in result
        assert "Added tomatoes to your inventory: 50 kg at 25 ETB per kg" in result["response"]
        assert "expires October 23, 2025" in result["response"]
        assert "deliverable Monday to Friday" in result["response"]
        # Should clear pending product
        assert "pending_product" not in result["session_context"]

    @pytest.mark.asyncio
    async def test_supplier_update_inventory(self, agent, mock_intent_classifier, mock_database_tool):
        """Test supplier updating existing inventory."""
        mock_intent_classifier.run.return_value = {
            "intent": "intent.supplier.update_inventory",
            "flow": "supplier",
            "filled_slots": {"product_name": "tomatoes", "quantity": 25},
            "missing_slots": [],
            "suggested_tools": []
        }

        # Mock database calls: find product, check existing supplier products, update
        mock_database_tool.run.side_effect = [
            {"product_id": 1, "product_name_en": "tomatoes"},  # find_product_by_any_name
            [{  # Existing supplier products
                "inventory_id": 10,
                "quantity_available": 50,
                "unit_price_etb": 25.0,
                "available_delivery_days": "Monday-Friday"
            }],
            None  # Update success
        ]

        session_context = {"user_id": 2}
        result = await agent.process_message("Add 25 kg more tomatoes to my inventory", session_context)

        assert "response" in result
        assert "Added 25 kg to your existing tomatoes inventory" in result["response"]
        assert "Total quantity now: 75 kg" in result["response"]
        assert "at 25.0 ETB per kg" in result["response"]