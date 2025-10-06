from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import os
from dotenv import load_dotenv
import logging
import re
import urllib.parse
from pipelinefinal import Pipeline
import json
from datetime import datetime
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Pipeline
try:
    pipeline = Pipeline(
        os.environ["GROQ_API_KEY"],
        "Policy2_Flexible_Full.pdf",
        os.environ["FIREBASE_CREDENTIALS_PATH"],
        "conversational-buying-assitant.firebasestorage.app"
    )
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pipeline: {str(e)}")
    raise

# Initialize Slack app
app = App(token=os.environ["SLACK_TOKEN"])

# Temporary storage for user preferences, welcome message tracking, pending queries, and carts (session only)
user_preferences = {}
welcome_sent = {}  # Tracks if welcome message has been sent to avoid duplicates
pending_queries = {}  # Stores queries until preferences are set
user_carts = {}  # Stores user carts: {user_id: [{product: {...}, quantity: int, compliance_status: str}, ...]}


def split_text_for_slack(text, max_length=3000):
    """Split text into Slack-friendly chunks and auto-format section headers."""
    # Remove horizontal rules (---)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

    # Convert Markdown headers to Slack style
    text = re.sub(r"^### (.*)", r"_*\1*_", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*)", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.*)", r"_*\1*_", text, flags=re.MULTILINE)

    # Convert bullet lists (- or *) to •
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)

    # Add spacing before numbered lists for readability
    text = re.sub(r"(\d+\.\s+)", r"\n\1", text)

    # Auto-detect and bold lines that look like section headers (end with ":")
    def bold_headers(match):
        line = match.group(1).strip()
        return f"*{line}*"

    text = re.sub(r"^(?!•|\d+\.)\s*([A-Z][A-Za-z0-9'&\-\s]+:)\s*$", bold_headers, text, flags=re.MULTILINE)

    # Normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split into chunks under 3000 chars
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def parse_price(price_str):
    """Parse price string (e.g., '₹2,899') to float."""
    try:
        # Remove currency symbols and commas
        cleaned_price = re.sub(r'[^\d.]', '', price_str)
        return float(cleaned_price)
    except (ValueError, TypeError):
        logger.error(f"Failed to parse price: {price_str}")
        return 0.0

def generate_purchase_order(user_id):
    """Generate a purchase order JSON for approved or recommended items in the cart."""
    cart = user_carts.get(user_id, [])
    if not cart:
        return None

    # Filter items with "Recommended" or "Approved" status
    valid_items = [item for item in cart if item["compliance_status"] in ["Recommended", "Approved"]]
    if not valid_items:
        return None

    # Group items by vendor_source (product source)
    vendor_groups = {}
    for item in valid_items:
        product = item["product"]
        vendor = product.get('source', 'Unknown')
        if vendor not in vendor_groups:
            vendor_groups[vendor] = []
        amount_each = parse_price(product.get('price', '0'))
        if amount_each == 0.0:
            logger.warning(f"Skipping item {product.get('title', 'Unknown')} due to invalid price")
            continue
        vendor_groups[vendor].append({
            "quantity": item["quantity"],
            "description": product.get('title', 'Unknown Product'),
            "amount_each": amount_each,
            "total_amount": amount_each * item["quantity"]
        })

    # Create the purchase order JSON
    po = {
        "document_type": "Purchase Order",
        "document_date": datetime.now().strftime("%d-%m-%Y"),
        "document_language": "English",
        "currency": "INR",
        "order_number": f"PO{int(datetime.now().timestamp())}",  # Simple unique order number
        "lines": [
            {"vendor_source": vendor, "line_items": vendor_items}
            for vendor, vendor_items in vendor_groups.items()
        ]
    }
    return po

def build_product_blocks(products):
    blocks = []
    for product in products[:10]:  # Limit to 10 to avoid Slack's 50-block limit
        title = product.get('title', '').strip()
        product_link = product.get('product_link', '')
        product_link = urllib.parse.quote(product_link, safe=':/?&=,')
        product_link = product_link.replace('|', '%7C')
        product_id = product.get('id', f'prod_{hash(title)}')

        blocks.extend([
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"<{product_link}|{title}>\n"
                        f"*Price:* {product['price']} | *Source:* {product['source']}\n"
                        f"*Rating:* {product['rating']} ⭐ | *Reviews:* {product['reviews']}"
                    )
                },
                "accessory": {
                    "type": "image",
                    "image_url": product['image'],
                    "alt_text": "Product Image"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "static_select",
                        "placeholder": {"type": "plain_text", "text": "Qty"},
                        "options": [
                            {"text": {"type": "plain_text", "text": str(i)}, "value": str(i)}
                            for i in range(1, 11)
                        ] + [
                            {"text": {"type": "plain_text", "text": "Custom quantity"}, "value": "custom"}
                        ],
                        "action_id": f"select_qty_{product_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Add to Cart"},
                        "style": "primary",
                        "value": f"{product_id}|1",
                        "action_id": f"add_to_cart_{product_id}"
                    }
                ]
            },
            {"type": "divider"}
        ])

    return blocks if blocks else [{
        "type": "section",
        "text": {"type": "mrkdwn", "text": "No products found. Please try a different query."}
    }]

def build_cart_blocks(user_id):
    cart = user_carts.get(user_id, [])
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": ":shopping_trolley: *Your Shopping Cart*"}
        },
        {"type": "divider"}
    ]

    if not cart:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "Your cart is empty."}
        })
    else:
        for item in cart:
            product = item["product"]
            quantity = item["quantity"]
            compliance_status = item.get("compliance_status", "In cart")
            title = product.get('title', '').strip()
            product_link = product.get('product_link', '')
            product_link = urllib.parse.quote(product_link, safe=':/?&=,')
            product_link = product_link.replace('|', '%7C')
            product_id = product.get('id', f'prod_{hash(title)}')

            # Format status with color indicators
            status_text = compliance_status
            if compliance_status == "Recommended":
                status_text = f"*Recommended*"
            elif compliance_status == "Awaiting Approval":
                status_text = f"_Awaiting Approval_"
            elif compliance_status == "Non Compliant":
                status_text = f"*Non-Compliant*"
            elif compliance_status == "Approved":
                status_text = f"*Approved*"
            elif compliance_status == "Rejected":
                status_text = f"*Rejected*"
            elif compliance_status == "In cart":
                status_text = "In cart"

            blocks.extend([
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"<{product_link}|{title}>\n"
                            f"*Price:* {product['price']} | *Source:* {product['source']}\n"
                            f"*Quantity:* {quantity}\n"
                            f"*Status:* {status_text}"
                        )
                    },
                    "accessory": {
                        "type": "image",
                        "image_url": product['image'],
                        "alt_text": "Product Image"
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Check Compliance"},
                            "action_id": f"check_compliance_{product_id}",
                            "value": f"{product_id}"
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Remove from Cart"},
                            "style": "danger",
                            "action_id": f"remove_from_cart_{product_id}",
                            "value": f"{product_id}"
                        }
                    ]
                },
                {"type": "divider"}
            ])

        # Add "Check Compliance for all" and "Proceed to PO creation" as a separate actions block
        blocks.extend([
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": " "},  # left column intentionally left blank
                    {"type": "mrkdwn", "text": "             "}  # adds spacing on right (em-spaces)
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Check Compliance for all"},
                        "action_id": "check_all_compliance",
                        "value": f"{user_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Proceed to PO creation"},
                        "style": "primary",
                        "action_id": "proceed_to_po",
                        "value": f"{user_id}"
                    }
                ]
            }
        ])

    return blocks
def open_preferences_modal(client, trigger_id, user_id):
    logger.debug(f"Attempting to open modal for user {user_id} with trigger_id {trigger_id}")
    try:
        client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": "preferences_submission",
                "title": {"type": "plain_text", "text": "Set Preferences and Role"},
                "submit": {"type": "plain_text", "text": "Save"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Select your preferences for product searches:"}
                    },
                    {
                        "type": "input",
                        "block_id": "preferences_block",
                        "element": {
                            "type": "multi_static_select",
                            "placeholder": {"type": "plain_text", "text": "Choose your preferences and let us know what kind of a user you are..."},
                            "options": [
                                {"text": {"type": "plain_text", "text": "Rating Conscious"}, "value": "rating_conscious"},
                                {"text": {"type": "plain_text", "text": "Price Conscious"}, "value": "price_conscious"},
                                {"text": {"type": "plain_text", "text": "Review Conscious"}, "value": "review_conscious"}
                            ],
                            "action_id": "preferences_select"
                        },
                        "label": {"type": "plain_text", "text": "Preferences"}
                    },
                    {
                        "type": "input",
                        "block_id": "role_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "role_input",
                            "placeholder": {"type": "plain_text", "text": "e.g., Junior Developer, Senior Engineer, Manager, Director, Executive"}
                        },
                        "label": {"type": "plain_text", "text": "Enter your role"}
                    }
                ]
            }
        )
        logger.debug(f"Modal opened successfully for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to open modal for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not open preferences modal. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not open preferences modal. Please try again."}
            }]
        )

def open_custom_qty_modal(client, trigger_id, user_id, product_id, product, channel_id, message_ts):
    logger.debug(f"Opening custom quantity modal for user {user_id}, product {product_id}")
    try:
        client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": f"custom_qty_submission_{product_id}",
                "private_metadata": f"{user_id}|{product_id}|{channel_id}|{message_ts}",
                "title": {"type": "plain_text", "text": "Enter Quantity"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "custom_qty_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "custom_qty_input",
                            "placeholder": {"type": "plain_text", "text": "Enter a number (e.g., 15)"}
                        },
                        "label": {"type": "plain_text", "text": f"Quantity for {product['title']}"}
                    }
                ]
            }
        )
        logger.debug(f"Custom quantity modal opened for user {user_id}")
    except Exception as e:
        logger.error(f"Failed to open custom quantity modal for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not open custom quantity modal. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not open custom quantity modal. Please try again."}
            }]
        )

@app.action({"action_id": re.compile(r"check_compliance_.*")})
def handle_check_compliance(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    trigger_id = body["trigger_id"]
    action = body["actions"][0]
    product_id = action["value"]

    # Find the product in the cart
    product = None
    cart_item_index = None
    for idx, item in enumerate(user_carts.get(user_id, [])):
        if item["product"].get('id', f'prod_{hash(item["product"]["title"])}') == product_id:
            product = item["product"]
            cart_item_index = idx
            break

    if not product:
        logger.error(f"Product not found for product_id {product_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Product not found in cart. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found in cart. Please try again."}
            }]
        )
        return

    try:
        # Get user role
        user_role = user_preferences.get(user_id, {}).get("role", "Unknown")
        price = parse_price(product.get('price', '0'))
        compliance_result = pipeline.compliance_checker.check_compliance(
            user_role=user_role,
            product_name=product.get('title', 'Unknown Product'),
            product_price=price,
            vendor=product.get('source', 'Unknown')
        )

        # Extract Compliance Status
        match = re.search(r"Compliance Status:[*\s]+(\w+(?:-\w+)?)", compliance_result)
        compliance_status = match.group(1) if match else "Unknown"

        # Update compliance status in cart
        if cart_item_index is not None:
            user_carts[user_id][cart_item_index]["compliance_status"] = (
                "Recommended" if compliance_status == "Compliant" else
                "Awaiting Approval" if compliance_status == "Needs Approval" else
                "Non Compliant"
            )

        # Format status for modal
        status_text = compliance_status
        if compliance_status == "Compliant":
            status_text = f"*Compliant*"
        elif compliance_status == "Needs Approval":
            status_text = f"_Needs Approval_"
        elif compliance_status == "Non-Compliant":
            status_text = f"*Non-Compliant*"

        # Format compliance result
        formatted_result = compliance_result.replace("**", "*").strip()
        text_chunks = split_text_for_slack(formatted_result, max_length=3000)

        modal_blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Compliance Status for {product['title']}*\n{status_text}"}
            },
            {"type": "divider"}
        ]
        for chunk in text_chunks:
            modal_blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": chunk}
            })

        if compliance_status in ["Needs Approval", "Non-Compliant"]:
            modal_blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Seek Approval"},
                        "action_id": f"seek_approval_{product_id}",
                        "value": f"{product_id}"
                    }
                ]
            })

        client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": f"compliance_result_{product_id}",
                "private_metadata": f"{user_id}|{product_id}",
                "title": {"type": "plain_text", "text": "Compliance Check"},
                "close": {"type": "plain_text", "text": "Close"},
                "blocks": modal_blocks
            }
        )

        # Update cart display
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{user_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(user_id)
            }
        )

    except Exception as e:
        logger.error(f"Error checking compliance for product {product_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not check compliance. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not check compliance. Please try again."}
            }]
        )

@app.action({"action_id": re.compile(r"seek_approval_.*")})
def handle_seek_approval(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    action = body["actions"][0]
    product_id = action["value"]

    # Find the product
    product = None
    for item in user_carts.get(user_id, []):
        if item["product"].get('id', f'prod_{hash(item["product"]["title"])}') == product_id:
            product = item["product"]
            break

    if not product:
        logger.error(f"Product not found for product_id {product_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Product not found in cart.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found in cart."}
            }]
        )
        return

    # Manager user ID (replace with actual ID if fixed)
    manager_user_id = "U09K388GH1P"  # TODO: replace with actual manager Slack user ID

    # Send approval request to manager
    product_title = product.get('title', '').strip()
    product_link = product.get('product_link', '')
    product_link = urllib.parse.quote(product_link, safe=':/?&=,')
    product_link = product_link.replace('|', '%7C')

    client.chat_postMessage(
        channel=manager_user_id,
        # Fallback text with link included
        text=f"Approval Request\nUser <@{user_id}> requests approval for:\n<{product_link}|{product_title}>\n*Price:* {product['price']} | *Source:* {product['source']}",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    # The primary block text with the link formatted for mrkdwn
                    "text": f"Approval Request\nUser <@{user_id}> requests approval for:\n<{product_link}|{product_title}>\n*Price:* {product['price']} | *Source:* {product['source']}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Accept"},
                        "style": "primary",
                        "value": f"{user_id}|{product_id}|accepted",
                        "action_id": f"approval_accept_{product_id}"
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "value": f"{user_id}|{product_id}|rejected",
                        "action_id": f"approval_reject_{product_id}"
                    }
                ]
            }
        ]
    )

    # Notify requester
    client.chat_postMessage(
        channel=user_id,
        text=f"Approval request for {product['title']} sent to your manager.",
        blocks=[{
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Approval request for {product['title']} sent to your manager."}
        }]
    )

@app.action({"action_id": re.compile(r"approval_accept_.*")})
def handle_approval_accept(ack, body, client, logger):
    ack()
    value = body["actions"][0]["value"]
    requester_id, product_id, response = value.split("|")
    manager_id = body["user"]["id"]

    # Update the requester cart status
    cart = user_carts.get(requester_id, [])
    product_title = None
    for item in cart:
        if item["product"].get('id', f'prod_{hash(item["product"]["title"])}') == product_id:
            item["compliance_status"] = "Approved"
            product_title = item["product"]["title"]
            break

    if product_title:
        # Notify the requester
        client.chat_postMessage(
            channel=requester_id,
            text=f"Your request for {product_title} was approved by <@{manager_id}>.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Your request for {product_title} was *approved* by <@{manager_id}>."}
            }]
        )

        # Update requester Home tab
        client.views_publish(
            user_id=requester_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{requester_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(requester_id)
            }
        )
    else:
        logger.error(f"Product {product_id} not found in cart for user {requester_id}")
        client.chat_postMessage(
            channel=requester_id,
            text="Error: Product not found in cart.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found in cart."}
            }]
        )

@app.action({"action_id": re.compile(r"approval_reject_.*")})
def handle_approval_reject(ack, body, client, logger):
    ack()
    value = body["actions"][0]["value"]
    requester_id, product_id, response = value.split("|")
    manager_id = body["user"]["id"]

    # Update the requester cart status
    cart = user_carts.get(requester_id, [])
    product_title = None
    for item in cart:
        if item["product"].get('id', f'prod_{hash(item["product"]["title"])}') == product_id:
            item["compliance_status"] = "Rejected"
            product_title = item["product"]["title"]
            break

    if product_title:
        # Notify the requester
        client.chat_postMessage(
            channel=requester_id,
            text=f"Your request for {product_title} was rejected by <@{manager_id}>.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Your request for {product_title} was *rejected* by <@{manager_id}>."}
            }]
        )

        # Update requester Home tab
        client.views_publish(
            user_id=requester_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{requester_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(requester_id)
            }
        )
    else:
        logger.error(f"Product {product_id} not found in cart for user {requester_id}")
        client.chat_postMessage(
            channel=requester_id,
            text="Error: Product not found in cart.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found in cart."}
            }]
        )

@app.action("check_all_compliance")
def handle_check_all_compliance(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    cart = user_carts.get(user_id, [])
    if not cart:
        client.chat_postMessage(
            channel=user_id,
            text="Your cart is empty. Nothing to check for compliance.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Your cart is empty. Nothing to check for compliance."}
            }]
        )
        return

    try:
        user_role = user_preferences.get(user_id, {}).get("role", "Unknown")
        results = []
        for idx, item in enumerate(cart):
            product = item["product"]
            price = parse_price(product.get('price', '0'))
            compliance_result = pipeline.compliance_checker.check_compliance(
                user_role=user_role,
                product_name=product.get('title', 'Unknown Product'),
                product_price=price,
                vendor=product.get('source', 'Unknown')
            )
            match = re.search(r"Compliance Status:[*\s]+(\w+(?:-\w+)?)", compliance_result)
            compliance_status = match.group(1) if match else "Unknown"
            user_carts[user_id][idx]["compliance_status"] = (
                "Recommended" if compliance_status == "Compliant" else
                "Awaiting Approval" if compliance_status == "Needs Approval" else
                "Non Compliant"
            )
            results.append(f"{product['title']}: {compliance_status}")

        # Notify user
        client.chat_postMessage(
            channel=user_id,
            text="Compliance check completed for all items.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Compliance Check Results*\n" + "\n".join([f"• {result}" for result in results])}
            }]
        )

        # Update cart display
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{user_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(user_id)
            }
        )

    except Exception as e:
        logger.error(f"Error checking compliance for all items for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not check compliance for all items. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not check compliance for all items. Please try again."}
            }]
        )

@app.action("proceed_to_po")
def handle_proceed_to_po(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    cart = user_carts.get(user_id, [])
    if not cart:
        client.chat_postMessage(
            channel=user_id,
            text="Your cart is empty. Nothing to proceed with.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Your cart is empty. Nothing to proceed with."}
            }]
        )
        return

    try:
        user_role = user_preferences.get(user_id, {}).get("role", "Unknown")
        # Check compliance for items with "In cart" status
        for idx, item in enumerate(cart):
            if item["compliance_status"] == "In cart":
                product = item["product"]
                price = parse_price(product.get('price', '0'))
                compliance_result = pipeline.compliance_checker.check_compliance(
                    user_role=user_role,
                    product_name=product.get('title', 'Unknown Product'),
                    product_price=price,
                    vendor=product.get('source', 'Unknown')
                )
                match = re.search(r"Compliance Status:[*\s]+(\w+(?:-\w+)?)", compliance_result)
                compliance_status = match.group(1) if match else "Unknown"
                user_carts[user_id][idx]["compliance_status"] = (
                    "Recommended" if compliance_status == "Compliant" else
                    "Awaiting Approval" if compliance_status == "Needs Approval" else
                    "Non Compliant"
                )

        # Generate purchase order
        po = generate_purchase_order(user_id)
        if not po:
            client.chat_postMessage(
                channel=user_id,
                text="No eligible items (Recommended or Approved) for purchase order creation.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "No eligible items (*Recommended* or *Approved*) for purchase order creation."}
                }]
            )
            return

        # Upload JSON file to Slack using modern external upload methods
        po_json = json.dumps(po, indent=2)
        filename = f"purchase_order_{po['order_number']}.json"
        content_length = len(po_json.encode('utf-8'))

        response1 = client.files_getUploadURLExternal(filename=filename, length=content_length)
        if not response1['ok']:
            raise Exception(f"Error getting upload URL: {response1['error']}")

        upload_url = response1['upload_url']
        file_id = response1['file_id']

        response2 = requests.post(upload_url, files={'file': (filename, po_json, 'application/json')})
        if response2.status_code != 200:
            raise Exception(f"Upload failed: {response2.status_code} - {response2.text}")

        response3 = client.files_completeUploadExternal(files=[{'id': file_id, 'title': "Purchase Order"}])
        if not response3['ok']:
            raise Exception(f"Error completing upload: {response3['error']}")

        file = response3['files'][0]
        permalink = file['permalink']

        client.chat_postMessage(
            channel=user_id,
            text=f"Here is your generated Purchase Order: {permalink}"
        )

        # Update cart display
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{user_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(user_id)
            }
        )

    except Exception as e:
        logger.error(f"Error generating purchase order for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not generate purchase order. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not generate purchase order. Please try again."}
            }]
        )

@app.action({"action_id": re.compile(r"remove_from_cart_.*")})
def handle_remove_from_cart(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    action = body["actions"][0]
    product_id = action["value"]

    # Find and remove the product from the cart
    product_title = None
    if user_id in user_carts:
        for idx, item in enumerate(user_carts[user_id]):
            if item["product"].get('id', f'prod_{hash(item["product"]["title"])}') == product_id:
                product_title = item["product"]["title"]
                user_carts[user_id].pop(idx)
                break

    if product_title:
        logger.debug(f"Removed product {product_id} from cart for user {user_id}")
        client.chat_postMessage(
            channel=user_id,
            text=f"Removed {product_title} from your cart.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Removed {product_title} from your cart."}
            }]
        )

        # Update cart display
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{user_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(user_id)
            }
        )
    else:
        logger.error(f"Product not found for product_id {product_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Product not found in cart. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found in cart. Please try again."}
            }]
        )

@app.action({"action_id": re.compile(r"select_qty_.*")})
def handle_quantity_selection(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    trigger_id = body["trigger_id"]
    action = body["actions"][0]
    product_id = action["action_id"].replace("select_qty_", "")
    selected_qty = action["selected_option"]["value"]
    channel_id = body["channel"]["id"]
    message_ts = body["message"]["ts"]

    product = None
    for block in body["message"]["blocks"]:
        if block["type"] == "section":
            product_link = block["text"]["text"].split("\n")[0].split("|")[0].strip("<>")
            for p in pipeline.last_results:
                if p.get('id', f'prod_{hash(p["title"])}') == product_id:
                    product = p
                    break
        if product:
            break

    if not product:
        logger.error(f"Product not found for product_id {product_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Product not found. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found. Please try again."}
            }]
        )
        return

    if selected_qty == "custom":
        open_custom_qty_modal(client, trigger_id, user_id, product_id, product, channel_id, message_ts)
    else:
        for block in body["message"]["blocks"]:
            if block["type"] == "actions" and block["elements"][1]["action_id"] == f"add_to_cart_{product_id}":
                block["elements"][1]["value"] = f"{product_id}|{selected_qty}"
                break

        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            blocks=body["message"]["blocks"]
        )
        logger.debug(f"Updated quantity to {selected_qty} for product {product_id} for user {user_id}")

@app.view(re.compile(r"custom_qty_submission_.*"))
def handle_custom_qty_submission(ack, body, client, view):
    ack()
    user_id, product_id, channel_id, message_ts = view["private_metadata"].split("|")
    custom_qty = view["state"]["values"]["custom_qty_block"]["custom_qty_input"]["value"]

    try:
        quantity = int(custom_qty)
        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        product = None
        for p in pipeline.last_results:
            if p.get('id', f'prod_{hash(p["title"])}') == product_id:
                product = p
                break

        if not product:
            logger.error(f"Product not found for product_id {product_id}")
            client.chat_postMessage(
                channel=user_id,
                text="Error: Product not found. Please try again.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Error: Product not found. Please try again."}
                }]
            )
            return

        message = client.conversations_history(channel=channel_id, latest=message_ts, limit=1, inclusive=True)
        blocks = message["messages"][0]["blocks"]

        for block in blocks:
            if block["type"] == "actions" and block["elements"][1]["action_id"] == f"add_to_cart_{product_id}":
                block["elements"][1]["value"] = f"{product_id}|{quantity}"
                break

        client.chat_update(
            channel=channel_id,
            ts=message_ts,
            blocks=blocks
        )
        logger.debug(f"Custom quantity {quantity} set for product {product_id} for user {user_id}")

    except ValueError:
        logger.error(f"Invalid quantity input '{custom_qty}' for user {user_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Please enter a valid positive number for quantity.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Please enter a valid positive number for quantity."}
            }]
        )
    except Exception as e:
        logger.error(f"Error updating message for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not update quantity. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not update quantity. Please try again."}
            }]
        )

@app.action({"action_id": re.compile(r"add_to_cart_.*")})
def handle_add_to_cart(ack, body, client):
    ack()
    user_id = body["user"]["id"]
    action = body["actions"][0]
    product_id, quantity = action["value"].split("|")
    quantity = int(quantity)

    product = None
    for p in pipeline.last_results:
        if p.get('id', f'prod_{hash(p["title"])}') == product_id:
            product = p
            break

    if not product:
        logger.error(f"Product not found for product_id {product_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Product not found. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Product not found. Please try again."}
            }]
        )
        return

    if user_id not in user_carts:
        user_carts[user_id] = []

    user_carts[user_id].append({"product": product, "quantity": quantity, "compliance_status": "In cart"})
    logger.debug(f"Added {quantity} of product {product_id} to cart for user {user_id}")

    client.chat_postMessage(
        channel=user_id,
        text=f"Added {quantity} x {product['title']} to your cart.",
        blocks=[{
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"Added {quantity} x {product['title']} to your cart."}
        }]
    )

@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    user_id = event["user"]
    try:
        client.views_publish(
            user_id=user_id,
            view={
                "type": "home",
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Hi <@{user_id}>! :wave:\n\nWelcome to our *Conversational Buying Space*. Meet your *Buying Bot*, designed to simplify and accelerate your entire purchasing workflow—all without leaving Slack.\n\n*GETTING STARTED (MESSAGES TAB)*\n1. *Set Up*: To ensure accurate pricing and compliance, you must set your *Role* and *Preferences* (e.g., price-conscious). You will be prompted on your first query, or click the Set Preferences button.\n2. *Request*: In the Messages tab, simply chat your product need (e.g., “@BuyingBot need a new mouse”).\n3. *Select*: I'll display personalized product cards. Use the Quantity selector and click Add to Cart on the products you want.\n\n*MANAGE AND APPROVE (HOME TAB)*\nThis Home tab is your central hub for managing procurement actions:\n\n- :one: *Review Cart*\n  • *Action*: Check the cart below. Status is initially *In cart*.\n  • *Next Step*: Click Check Compliance next to an item.\n- :two: *Compliance Check*\n  • *Action*: Click Check Compliance or Check Compliance for all.\n  • *Result*: Status updates to *Recommended*, *Needs Approval*, or *Non-Compliant*.\n- :three: *Seek Approval*\n  • *Action*: If status is *Needs Approval*, click the button to notify your manager.\n  • *Result*: Status updates to *Approved* or *Rejected* based on their action.\n- :four: *PO Creation*\n  • *Action*: Once items are *Recommended* or *Approved*, click Proceed to PO creation.\n  • *Result*: I'll generate and deliver the final Purchase Order (PO) file here!\n\n*READY TO SHOP SMARTER*? Explore your Shopping Cart below, or switch to the Messages tab and type away your requests!"}
                    },
                    {"type": "divider"}
                ] + build_cart_blocks(user_id)
            }
        )
        logger.debug(f"Updated home tab with welcome message and cart for user {user_id}")
        welcome_sent[user_id] = True
    except Exception as e:
        logger.error(f"Error updating home tab for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not update Home tab. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not update Home tab. Please try again."}
            }]
        )

@app.view("preferences_submission")
def handle_preferences_submission(ack, body, client, view):
    logger.debug(f"Handling preferences submission for user {body['user']['id']}")
    try:
        ack()
        user_id = body["user"]["id"]
        selected_preferences = view["state"]["values"]["preferences_block"]["preferences_select"]["selected_options"]
        role = view["state"]["values"]["role_block"]["role_input"]["value"].strip()
        preferences = [option["value"] for option in selected_preferences]

        user_preferences[user_id] = {"preferences": preferences, "role": role}
        pipeline.set_preferences(preferences)

        client.chat_postMessage(
            channel=user_id,
            text=f"Preferences selected: {', '.join(preferences) or 'None'}, Role: {role}",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Preferences selected: {', '.join(preferences) or 'None'}\n*Role:* {role}"}
            }]
        )
        logger.debug(f"Preferences saved for user {user_id}: {preferences}, Role: {role}")

        if user_id in pending_queries:
            query = pending_queries[user_id]
            try:
                response = pipeline.run(query)
                if isinstance(response, str):
                    formatted_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                    formatted_response = formatted_response.replace("**", "*").replace("★", ":star:")
                    text_chunks = split_text_for_slack(formatted_response, max_length=3000)

                    blocks = [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": ":sparkles: *Personalized Recommendations Summary*"}
                        },
                        {"type": "divider"}
                    ]
                    for chunk in text_chunks:
                        blocks.append({
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": chunk}
                        })

                    client.chat_postMessage(
                        channel=user_id,
                        text="Personalized Recommendations Summary",
                        blocks=blocks,
                        unfurl_links=False,
                        unfurl_media=False
                    )

                else:
                    logger.debug(f"Pipeline returned {len(response)} products for user {user_id}")
                    blocks = build_product_blocks(response)
                    client.chat_postMessage(
                        channel=user_id,
                        text=f"Found {len(response)} products matching your query.",
                        blocks=blocks,
                        unfurl_links=False,
                        unfurl_media=False
                    )

                    logger.debug(f"Displayed {len(response)} products for user {user_id}")

                del pending_queries[user_id]
            except Exception as e:
                logger.error(f"Error processing pending query '{query}' for user {user_id}: {str(e)}")
                client.chat_postMessage(
                    channel=user_id,
                    text=f"Error: Could not process your query '{query}'. Please try again or check your query format.",
                    blocks=[{
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"Error: Could not process your query '{query}'. Please try again or check your query format."}
                    }]
                )
    except Exception as e:
        logger.error(f"Error in preferences submission for user {user_id}: {str(e)}")
        client.chat_postMessage(
            channel=user_id,
            text="Error: Could not save preferences. Please try again.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Error: Could not save preferences. Please try again."}
            }]
        )

@app.action("open_preferences_button")
def handle_preferences_button(ack, body, client):
    ack()
    user_id = body["user"]["id"]
    trigger_id = body["trigger_id"]
    logger.debug(f"Preferences button clicked by user {user_id}")
    open_preferences_modal(client, trigger_id, user_id)

@app.event("app_mention")
def handle_mention(event, say, client, body):
    user_id = event["user"]
    query = event["text"].replace(f"<@{body['authorizations'][0]['user_id']}>", "").strip()
    logger.debug(f"Received app_mention from user {user_id}: {query}")

    if user_id not in welcome_sent:
        try:
            client.chat_postMessage(
                channel=user_id,
                text="Hi! I'm your Conversational Buying Assistant. Set your preferences to start shopping!",
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Tired of jumping between systems?\nI'm your Conversational Buying Assistant, and I'm here to streamline your workflow!\n\nI turn requesting, justifying, and approving purchases into a simple chat—all right here in Slack. No more friction, just smart, fast buying.\n\nFor a quick introduction to my features and capabilities, just click the 'Home' tab above. It's the best place to learn about the process and manage your Shopping Cart. Ready to save time? Set your preferences and role, then just type away your requests!"}
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Set Preferences"},
                                "action_id": "open_preferences_button",
                                "style": "primary"
                            }
                        ]
                    }
                ]
            )
            welcome_sent[user_id] = True
            logger.debug(f"Sent welcome message with button to user {user_id}")
        except Exception as e:
            logger.error(f"Error sending welcome message to user {user_id}: {str(e)}")
            client.chat_postMessage(
                channel=user_id,
                text="Error: Could not send welcome message. Please try again.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Error: Could not send welcome message. Please try again."}
                }]
            )

    if user_id in user_preferences:
        try:
            response = pipeline.run(query)
            if isinstance(response, str):
                formatted_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                formatted_response = formatted_response.replace("**", "*").replace("★", ":star:")
                text_chunks = split_text_for_slack(formatted_response, max_length=3000)

                blocks = [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": ":sparkles: *Personalized Recommendations Summary*"}
                    },
                    {"type": "divider"}
                ]
                for chunk in text_chunks:
                    blocks.append({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": chunk}
                    })

                client.chat_postMessage(
                    channel=user_id,
                    text="Personalized Recommendations Summary",
                    blocks=blocks,
                    unfurl_links=False,
                    unfurl_media=False
                )

            else:
                logger.debug(f"Pipeline returned {len(response)} products for user {user_id}")
                blocks = build_product_blocks(response)
                client.chat_postMessage(
                    channel=user_id,
                    text=f"Found {len(response)} products matching your query.",
                    blocks=blocks,
                    unfurl_links=False,
                    unfurl_media=False
                )

                logger.debug(f"Displayed {len(response)} products for user {user_id}")

        except Exception as e:
            logger.error(f"Error processing query '{query}' for user {user_id}: {str(e)}")
            client.chat_postMessage(
                channel=user_id,
                text=f"Error: Could not process your query '{query}'. Please try again or check your query format.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"Error: Could not process your query '{query}'. Please try again or check your query format."}
                }]
            )
    else:
        pending_queries[user_id] = query
        logger.debug(f"Stored pending query '{query}' for user {user_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Please set your preferences before I can process your query. Click 'Set Preferences' to continue.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Please set your preferences before I can process your query. Click 'Set Preferences' to continue."}
            }]
        )

@app.message()
def handle_message(message, say, client, body):
    user_id = message["user"]
    query = message["text"]
    logger.debug(f"Received message from user {user_id}: {query}")

    if user_id not in welcome_sent:
        try:
            client.chat_postMessage(
                channel=user_id,
                text="Hi! I'm your Conversational Buying Assistant. Set your preferences to start shopping!",
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "Tired of jumping between systems?\nI'm your Conversational Buying Assistant, and I'm here to streamline your workflow!\n\nI turn requesting, justifying, and approving purchases into a simple chat—all right here in Slack. No more friction, just smart, fast buying.\n\nFor a quick introduction to my features and capabilities, just click the'Home' tab above. It's the best place to learn about the process and manage your Shopping Cart. Ready to save time? Set your preferences and role, then just type away your requests!"}
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Set Preferences"},
                                "action_id": "open_preferences_button",
                                "style": "primary"
                            }
                        ]
                    }
                ]
            )
            welcome_sent[user_id] = True
            logger.debug(f"Sent welcome message with button to user {user_id}")
        except Exception as e:
            logger.error(f"Error sending welcome message to user {user_id}: {str(e)}")
            client.chat_postMessage(
                channel=user_id,
                text="Error: Could not send welcome message. Please try again.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "Error: Could not send welcome message. Please try again."}
                }]
            )

    if user_id in user_preferences:
        try:
            logger.debug(f"Running pipeline.run with query: {query}")
            response = pipeline.run(query)
            if isinstance(response, str):
                formatted_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                formatted_response = formatted_response.replace("**", "*").replace("★", ":star:")
                text_chunks = split_text_for_slack(formatted_response, max_length=3000)

                blocks = [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": ":sparkles: *Personalized Recommendations Summary*"}
                    },
                    {"type": "divider"}
                ]
                for chunk in text_chunks:
                    blocks.append({
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": chunk}
                    })

                client.chat_postMessage(
                    channel=user_id,
                    text="Personalized Recommendations Summary",
                    blocks=blocks,
                    unfurl_links=False,
                    unfurl_media=False
                )

            else:
                logger.debug(f"Pipeline returned {len(response)} products for user {user_id}")
                blocks = build_product_blocks(response)
                client.chat_postMessage(
                    channel=user_id,
                    text=f"Found {len(response)} products matching your query.",
                    blocks=blocks,
                    unfurl_links=False,
                    unfurl_media=False
                )

                logger.debug(f"Displayed {len(response)} products for user {user_id}")

        except Exception as e:
            logger.error(f"Error processing query '{query}' for user {user_id}: {str(e)}")
            client.chat_postMessage(
                channel=user_id,
                text=f"Error: Could not process your query '{query}'. Please try again or check your query format.",
                blocks=[{
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"Error: Could not process your query '{query}'. Please try again or check your query format."}
                }]
            )
    else:
        pending_queries[user_id] = query
        logger.debug(f"Stored pending query '{query}' for user {user_id}")
        client.chat_postMessage(
            channel=user_id,
            text="Please set your preferences before I can process your query. Click 'Set Preferences' to continue.",
            blocks=[{
                "type": "section",
                "text": {"type": "mrkdwn", "text": "Please set your preferences before I can process your query. Click 'Set Preferences' to continue."}
            }]
        )

if __name__ == "__main__":
    logger.info("Starting Slack Bolt app")
    handler = SocketModeHandler(app, os.environ["APP_TOKEN"])
    handler.start()
