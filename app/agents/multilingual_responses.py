"""
Multilingual response dictionary for KCartBot agent.
Contains translations for all user-facing messages in English, Amharic, and Phonetic Amharic.
"""

from typing import Dict


def get_multilingual_response_dictionary() -> Dict[str, Dict[str, str]]:
    """Get the complete multilingual response dictionary."""
    return {
        # Greetings and general
        "greeting": {
            "english": "Hello! Welcome to KCartBot, your fresh produce marketplace assistant. Are you a customer looking to place an order, or a supplier managing inventory?",
            "amharic": "ሰላም! ወደ KCartBot እንኳን ደህና መጡ - የእርስዎ የአዲስ ምርት ገበያ ረዳት። ትዕዛዝ ለመስጠት የሚፈልጉ ደንበኛ ነዎት ወይስ አቅራቢ?",
            "phonetic_amharic": "Selam! Wede KCartBot enkuan dehna metu - yerso ye'adis miret gebeya redat. Tizaz lemesṭet yemīfeligu denbeñña nehot weyis aḳrabī?"
        },
        "unknown_intent": {
            "english": "I'm not sure what you mean. Are you a customer looking to place an order, or a supplier managing inventory?",
            "amharic": "ምን ማለትዎ እንደሆነ አልገባኝም። ትዕዛዝ ለመስጠት የሚፈልጉ ደንበኛ ነዎት ወይስ አቅራቢ?",
            "phonetic_amharic": "Min maletwo endehone algebagnim. Tizaz lemesṭet yemīfeligu denbeñña nehot weyis aḳrabī?"
        },
        "confirmation_response": {
            "english": "Great! Could you please provide more details about what you'd like to do?",
            "amharic": "በጣም ጥሩ! ስለሚፈልጉት ነገር ተጨማሪ ዝርዝር ሊሰጡኝ ይችላሉ?",
            "phonetic_amharic": "Betam t'iru! Silemmifeligut neger t'emari zirzir liset'ugn yichilalu?"
        },

        # Onboarding
        "is_customer": {
            "english": "Great! Are you a new customer or do you already have an account with us?",
            "amharic": "እባክዎ! አዲስ ተጠቃሚ ነን ወይም ከዚህ ቀድሞ ተጠቃሚ ነን?",
            "phonetic_amharic": "Ey bekual! Adis temekemi nen weyim kezzih kedmo temekemi nen?"
        },
        "is_supplier": {
            "english": "Excellent! Are you a new supplier or do you already have an account with us?",
            "amharic": "በጣም ጥሩ! አዲስ አቅራቢ ነዎት ወይስ ከእኛ ጋር አካውንት አለዎት?",
            "phonetic_amharic": "Betam t'iru! Addis ak'rabi nehot weyis ke'igna gar akaunt alewot?"
        },
        "has_account": {
            "english": "I see you already have an account. What's your name and phone number so I can verify your {user_role} account?",
            "amharic": "አካውንት እንዳለዎት አይቻለሁ። የእርስዎን {user_role} አካውንት ለማረጋገጥ ስምዎን እና ስልክ ቁጥርዎን ሊነግሩኝ ይችላሉ?",
            "phonetic_amharic": "Akaunt endalewot ayichalehu. Yersuwan {user_role} akaunt lemaregaget simiwon ina silik k'ut'riwon linegruñ yichilalu?"
        },
        "new_user": {
            "english": "Great! Let's get you registered. What's your name?",
            "amharic": "በጣም ጥሩ! እስቲ ይመዝገቡ። ስምዎ ማን ይባላል?",
            "phonetic_amharic": "Betam t'iru! Isti yimezgebu. Simwo man yibalal?"
        },

        # Registration
        "ask_customer_name": {
            "english": "Welcome! What's your name?",
            "amharic": "እንኳን ደህና መጡ! ስምዎ ማን ነው?",
            "phonetic_amharic": "Enkuan dehna metu! Simwo man new?"
        },
        "ask_phone_number": {
            "english": "Great! What's your phone number?",
            "amharic": "በጣም ጥሩ! ስልክ ቁጥርዎ ስንት ነው?",
            "phonetic_amharic": "Betam t'iru! Silik k'ut'irwo sint new?"
        },
        "ask_default_location": {
            "english": "Perfect! What's your default delivery location?",
            "amharic": "ምርጥ! ስትሪም የማስረከቢያ ቦታዎ የት ነው?",
            "phonetic_amharic": "Mirt'! Stiram yemaskerebiya botawo yet new?"
        },
        "ask_business_name": {
            "english": "Welcome! What's your business name?",
            "amharic": "እንኳን ደህና መጡ! የንግድ ስምዎ ማን ነው?",
            "phonetic_amharic": "Enkuan dehna metu! Yenigid simwo man new?"
        },

        # Verification
        "need_name_phone": {
            "english": "I need both your name and phone number to verify your account.",
            "amharic": "አካውንትዎን ለማረጋገጥ ስምዎ እና ስልክ ቁጥርዎ ያስፈልጉኛል።",
            "phonetic_amharic": "Akauntiwon lemaregaget simwo ina silik k'ut'irwo yasfeligugnal."
        },
        "account_verified_customer": {
            "english": "Welcome back, {user_name}! Your customer account has been verified. How can I help you with your fresh produce needs today?",
            "amharic": "እንኳን ደህና መመለስዎ, {user_name}! የደንበኛ አካውንትዎ ተረጋግጧል። ዛሬ በአዲስ ምርት ፍላጎቶችዎ እንዴት ልረዳዎት እችላለሁ?",
            "phonetic_amharic": "Enkuan dehna memeleswo, {user_name}! Yedenbeñña akauntwo teregagt'wal. Zarē be'ādis miriti filagotowo indēt liredawot ichilalehu?"
        },
        "account_verified_supplier": {
            "english": "Welcome back, {user_name}! Your supplier account has been verified. {dashboard_info} How can I help you manage your inventory today?",
            "amharic": "እንኳን ደህና መመለስዎ, {user_name}! የአቅራቢነት አካውንትዎ ተረጋግጧል። {dashboard_info} ዛሬ ክምችትዎን ለማስተዳደር እንዴት ልረዳዎት እችላለሁ?",
            "phonetic_amharic": "Enkuan dehna memeleswo, {user_name}! Ye'āk'rabīnet akauntwo teregagt'wal. {dashboard_info} Zarē kimichitowon lemastedader indēt liredawot ichilalehu?"
        },
        "account_not_found": {
            "english": "I couldn't find an account with that name and phone number. Would you like to create a new {user_role} account instead?",
            "amharic": "በዚህ ስም እና ስልክ ቁጥር አካውንት ማግኘት አልቻልኩም። በምትኩ አዲስ የ{user_role} አካውንት መክፈት ይፈልጋሉ?",
            "phonetic_amharic": "Bezih sim ina silik k'ut'ir akaunt magignet alchalkum. Bemitiku addis ye{user_role} akaunt mekifet yifeligalu?"
        },

        # Registration success
        "customer_registered": {
            "english": "Welcome {customer_name}! Your customer account has been created. How can I help you with your fresh produce needs today?",
            "amharic": "እንኳን ደህና መጡ {customer_name}! የደንበኛ አካውንትዎ ተፈጥሯል። ዛሬ በአዲስ ምርት ፍላጎቶችዎ እንዴት ልረዳዎት እችላለሁ?",
            "phonetic_amharic": "Enkuan dehna metu {customer_name}! Yedenbeñña akauntwo tefet'irwal. Zarē be'ādis miriti filagotowo indēt liredawot ichilalehu?"
        },
        "supplier_registered": {
            "english": "Welcome {supplier_name}! Your supplier account has been created. How can I help you manage your inventory today?",
            "amharic": "እንኳን ደህና መጡ {supplier_name}! የአቅራቢነት አካውንትዎ ተፈጥሯል። ዛሬ ክምችትዎን ለማስተዳደር እንዴት ልረዳዎት እችላለሁ?",
            "phonetic_amharic": "Enkuan dehna metu {supplier_name}! Ye'āk'rabīnet akauntwo tefet'irwal. Zarē kimichitowon lemastedader indēt liredawot ichilalehu?"
        },
        "registration_failed": {
            "english": "I couldn't create your account. Please try again.",
            "amharic": "ተጠቃሚ መፍጠር አልተሳካም። እንደገና ይሞክሩ።",
            "phonetic_amharic": "Akaunt mefṭer altesakalñim. Inidegena yimokiru."
        },

        # Product availability
        "what_product": {
            "english": "What product are you looking for?",
            "amharic": "ምን አይነት ምርት ነው የሚፈልጉት?",
            "phonetic_amharic": "Min aynet miriti new yemīfeligut?"
        },
        "product_available": {
            "english": "Yes, {product_name} is available from:",
            "amharic": "አዎ፣ {product_name} ከሚከተሉት ማቅረቢያዎች ይገኛል፡",
            "phonetic_amharic": "Awo, {product_name} kemīketelut āk'rabīwoch yigenyal:"
        },
        "product_not_available": {
            "english": "Sorry, {product_name} is not currently available.",
            "amharic": "ይቅርታ, {product_name} በአሁኑ ሰዓት አይገኝም።",
            "phonetic_amharic": "Yik'irta, {product_name} be'āhunu se'at ayigenyim."
        },
        "supplier_products": {
            "english": "Products available from {supplier_name}:",
            "amharic": "ከ{supplier_name} የሚገኙ ምርቶች፡",
            "phonetic_amharic": "Ke{supplier_name} yemīgenyut miritochi:"
        },
        "supplier_no_products": {
            "english": "Sorry, {supplier_name} doesn't have any products available right now.",
            "amharic": "ይቅርታ, {supplier_name} በአሁኑ ሰዓት ምንም አይነት ምርት የለውም።",
            "phonetic_amharic": "Yik'irta, {supplier_name} be'āhunu se'at minim aynet miriti yelewim."
        },

        # Storage advice
        "what_product_storage": {
            "english": "What product do you need storage advice for?",
            "amharic": "ለምን ምርት የማከማቻ ምክር ያስፈልግዎታል?",
            "phonetic_amharic": "Lemin miriti yemakemacha mikir yasfeligotal?"
        },

        # Nutrition queries
        "compare_products": {
            "english": "Which two products would you like to compare nutritionally?",
            "amharic": "በንጥረ ነገር ይዘታቸው የትኞቹን ሁለት ምርቶች ማወዳደር ይፈልጋሉ?",
            "phonetic_amharic": "Benit'ire neger yizetachew yet'iñochun hulet miritochi mawedader yifeligalu?"
        },
        "nutrition_not_found": {
            "english": "I couldn't find nutritional information comparing {product_a} and {product_b}.",
            "amharic": "{product_a} እና {product_b}ን የሚያወዳድር የአመጋገብ መረጃ ማግኘት አልቻልኩም።",
            "phonetic_amharic": "{product_a} ina {product_b}n yemīyawedadiri ye'āmigabi mereja maginet alchalkum."
        },

        # Seasonal queries
        "seasonal_not_found": {
            "english": "I don't have specific seasonal information right now.",
            "amharic": "በአሁኑ ሰዓት የተለየ የወቅታዊ መረጃ የለኝም።",
            "phonetic_amharic": "Be'āhunu se'at yeteleye yewok'itawī mereja yeleñim."
        },
        "seasonal_no_data": {
            "english": "I don't have specific seasonal information in my knowledge base right now.",
            "amharic": "በእውቀት ቋቴ ውስጥ በአሁኑ ሰዓት የተለየ የወቅታዊ መረጃ የለኝም።",
            "phonetic_amharic": "Be'iwik'eti k'wat'ē wisit'i be'āhunu se'at yeteleye yewok'itawī mereja yeleñim."
        },
        "seasonal_error": {
            "english": "I couldn't find seasonal availability information.",
            "amharic": "የወቅታዊ ተገኝነት መረጃ ማግኘት አልቻልኩም።",
            "phonetic_amharic": "Yewok'itawī teginyinet mereja maginet alchalkum."
        },
        "seasonal_fallback": {
            "english": "Seasonal information: {context}",
            "amharic": "ወቅታዊ መረጃ: {context}",
            "phonetic_amharic": "Wok'itawī mereja: {context}"
        },

        # General advisory
        "general_advisory_no_query": {
            "english": "I'd be happy to help with advice about fresh produce. What would you like to know?",
            "amharic": "ስለ አዲስ ምርት ምክር ለመስጠት ደስተኛ ነኝ። ምን ማወቅ ይፈልጋሉ?",
            "phonetic_amharic": "Sile addis miriti mikir lemesit'eti desiteñña neñi. Min mawok'i yifeligalu?"
        },
        "general_advisory_not_found": {
            "english": "I don't have information about that topic.",
            "amharic": "ስለዚህ ርዕስ መረጃ የለኝም።",
            "phonetic_amharic": "Silezih ri'isi mereja yeleñim."
        },
        "general_advisory_no_data": {
            "english": "I don't have information about that topic in my knowledge base right now.",
            "amharic": "በእውቀት ቋቴ ውስጥ ስለዚህ ርዕስ መረጃ የለኝም።",
            "phonetic_amharic": "Be'iwik'eti k'wat'ē wisit'i silezih ri'isi mereja yeleñim."
        },
        "general_advisory_fallback": {
            "english": "Here's some information: {context}",
            "amharic": "አንዳንድ መረጃ እነሆ፡ {context}",
            "phonetic_amharic": "Andandi mereja ineho: {context}"
        },
        "general_advisory_error": {
            "english": "I couldn't find information about that topic.",
            "amharic": "ስለዚህ ርዕስ መረጃ ማግኘት አልቻልኩም።",
            "phonetic_amharic": "Silezih ri'isi mereja maginet alchalkum."
        },

        # Order placement
        "insufficient_quantity": {
            "english": "Sorry, only {available_quantity} {unit} of {product_name} is available from the selected supplier.",
            "amharic": "ይቅርታ, ከተመረጠው አቅራቢ {available_quantity} {unit} {product_name} ብቻ ነው ያለው።",
            "phonetic_amharic": "Yik'irta, ketemereṭew āk'rabī {available_quantity} {unit} {product_name} bicha new yalew."
        },
        "user_not_found": {
            "english": "User not found. Please register first.",
            "amharic": "ተጠቃሚ አልተገኘም። እባክዎ መጀመሪያ ይመዝገቡ።",
            "phonetic_amharic": "Tetek'amī alitegenyem. Ibakwo mejemeriya yimezgebu."
        },
        "product_supplier_not_found": {
            "english": "Product or supplier information not found.",
            "amharic": "የምርት ወይም የአቅራቢ መረጃ አልተለመደም።",
            "phonetic_amharic": "Yemirit weyimi ye'āk'rabī mereja alitegenyem."
        },
        "what_order": {
            "english": "What would you like to order?",
            "amharic": "ምን ማዘዝ ይፈልጋሉ?",
            "phonetic_amharic": "Min mazez yifeligalu?"
        },
        "how_much": {
            "english": "How much would you like to order?",
            "amharic": "ምን ያህል ማዘዝ ይፈልጋሉ?",
            "phonetic_amharic": "Min yahil mazez yifeligalu?"
        },
        "when_delivery": {
            "english": "When would you like this delivered?",
            "amharic": "መቼ እንዲደርስዎት ይፈልጋሉ?",
            "phonetic_amharic": "Mechē indīderisiwoti yifeligalu?"
        },
        "order_placed": {
            "english": "Order placed successfully! Total: {total_price} ETB. Payment will be Cash on Delivery.",
            "amharic": "ትእዛዝ ተሳካ! ጠቅላላ ዋጋ፡ {total_price} ብር። ክፍያ በማቅረቢያ ላይ ይሆናል።",
            "phonetic_amharic": "Tizazwo beseriatu tekebilewal! Dmiri waga: {total_price} ETB. Kifiyaw bemak'irebiya gīzē yihonal."
        },
        "order_failed": {
            "english": "I couldn't place your order. Please try again.",
            "amharic": "ትእዛዝ መስጠት አልተሳካም። እንደገና ይሞክሩ።",
            "phonetic_amharic": "Tizaziwon masigebati alichalikumi. Inidegena yimokiru."
        },

        # Delivery date
        "when_delivery_date": {
            "english": "When would you like your delivery?",
            "amharic": "ማድረሻዎ መቼ እንዲሆን ይፈልጋሉ?",
            "phonetic_amharic": "Madirešawo mechē indīhoni yifeligalu?"
        },
        "delivery_date_set": {
            "english": "Delivery date set to {delivery_date}.",
            "amharic": "የማስረከቢያ ቀን ወደ {delivery_date} ተቀናብሯል።",
            "phonetic_amharic": "Yemasirekebīya k'eni wede {delivery_date} tek'inabirwal."
        },
        "delivery_date_failed": {
            "english": "I couldn't set the delivery date. Please try again.",
            "amharic": "የማስረከቢያውን ቀን ማዘጋጀት አልቻልኩም። እባክዎ እንደገና ይሞክሩ።",
            "phonetic_amharic": "Yemasirekebīyawini k'eni mazegajeti alichalikumi. Ibakwo inidegena yimokiru."
        },

        # Delivery location
        "where_delivery": {
            "english": "Where would you like your delivery?",
            "amharic": "የት እንዲደርስዎት ይፈልጋሉ?",
            "phonetic_amharic": "Yeti indīderisiwoti yifeligalu?"
        },
        "delivery_location_set": {
            "english": "Delivery location set to {location}.",
            "amharic": "የማስረከቢያ ቦታ ወደ {location} ተቀናብሯል።",
            "phonetic_amharic": "Yemasirekebīya bota wede {location} tek'inabirwal."
        },

        # Payment confirmation
        "which_order_payment": {
            "english": "Which order would you like to confirm payment for?",
            "amharic": "የትኛውን ትዕዛዝ ነው ክፍያውን ማረጋገጥ የሚፈልጉት?",
            "phonetic_amharic": "Yet'iñawini ti'izazi newi kifiyawini maregageti yemīfeliguti?"
        },
        "payment_confirmed": {
            "english": "Payment confirmed! Your order is now being prepared for delivery.",
            "amharic": "ክፍያ ተረጋገጠ! ትእዛዝዎ ለማቅረቢያ ተለካ።",
            "phonetic_amharic": "Kifiyaw teregagit'wal! Tizazwo ahun lemastagebiya yizegajali."
        },
        "payment_failed": {
            "english": "I couldn't confirm the payment. Please try again.",
            "amharic": "ክፍያ መረጋገጥ አልተሳካም። እንደገና ይሞክሩ።",
            "phonetic_amharic": "Kifiyawini maregageti alichalikumi. Inidegena yimokiru."
        },

        # Check deliveries
        "no_deliveries": {
            "english": "You have no deliveries scheduled.",
            "amharic": "ምንም የተቀጠረ ማድረሻ የለዎትም።",
            "phonetic_amharic": "Minimi yetek'et'ere madireša yelewotimi."
        },
        "no_deliveries_date": {
            "english": "You have no deliveries scheduled for {date}.",
            "amharic": "ለ {date} ምንም የተቀጠረ ማድረሻ የለዎትም።",
            "phonetic_amharic": "Le {date} minimi yetek'et'ere madireša yelewotimi."
        },
        "deliveries_failed": {
            "english": "I couldn't check your deliveries right now. Please try again.",
            "amharic": "በአሁኑ ሰዓት ማድረሻዎችዎን ማረጋገጥ አልቻልኩም። እባክዎ እንደገና ይሞክሩ።",
            "phonetic_amharic": "Be'āhunu se'at madirešawochiwoni maregageti alichalikumi. Ibakwo inidegena yimokiru."
        },

        # Supplier handlers
        "register_first": {
            "english": "Please register as a supplier first.",
            "amharic": "እባክዎ መጀመሪያ እንደ አቅራቢ ይመዝገቡ።",
            "phonetic_amharic": "Ibakwo mejemeriya inide āk'rabī yimezgebu."
        },
        "what_product_add": {
            "english": "What product would you like to add?",
            "amharic": "ምን አይነት ምርት መጨመር ይፈልጋሉ?",
            "phonetic_amharic": "Min aynet miriti mech'emeri yifeligalu?"
        },
        "what_quantity": {
            "english": "What's the quantity you have available?",
            "amharic": "ምን ያህል ብዛት አለዎት?",
            "phonetic_amharic": "Min yahili bizati alewoti?"
        },
        "what_price": {
            "english": "What's the price per kg in ETB?",
            "amharic": "በኪሎ ለኪሎ ዋጋ ስንት ብር ነው?",
            "phonetic_amharic": "Bekīlo wagaw sinti birr newi?"
        },
        "when_expiry": {
            "english": "When does this {product_name} expire? (You can say 'no expiry' if it doesn't expire)",
            "amharic": "ይህ {product_name} መቼ ነው የሚያልቀው? (የሚያበቃበት ቀን ከሌለው 'አያልቅም' ማለት ይችላሉ)",
            "phonetic_amharic": "Yih {product_name} mechē newi yemīyalk'ewi? (Yemīyabek'abeti k'eni kelēlewi 'ayalik'imi' maleti yichilalu)"
        },
        "what_delivery_days": {
            "english": "What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')",
            "amharic": "{product_name}ን በየትኞቹ ቀናት ማድረስ ይችላሉ? (ለምሳሌ 'ከሰኞ እስከ አርብ' ወይም 'የሳምንቱ ቀናት')",
            "phonetic_amharic": "{product_name}ni beyet'iñochu k'enati madireši yichilalu? (Lemisalē 'kesenyo isike aribi' weyimi 'yesaminitu k'enati')"
        },
        "what_expiry_date": {
            "english": "When does {product_name} expire? (You can say 'no expiry' if it doesn't expire)",
            "amharic": "{product_name} መቼ ነው የሚያልቀው? (የሚያበቃበት ቀን ከሌለው 'አያልቅም' ማለት ይችላሉ)",
            "phonetic_amharic": "{product_name} mechē newi yemīyalk'ewi? (Yemīyabek'abeti k'eni kelēlewi 'ayalik'imi' maleti yichilalu)"
        },
        "expiry_date_set": {
            "english": "Expiry date set to {date}. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')",
            "amharic": "የማበቃበት ቀን ወደ {date} ተቀናብሯል። {product_name}ን በየትኞቹ ቀናት ማድረስ ይችላሉ? (ለምሳሌ 'ከሰኞ እስከ አርብ' ወይም 'የሳምንቱ ቀናት')",
            "phonetic_amharic": "Yemabek'abeti k'eni wede {date} tek'inabirwal. {product_name}ni beyet'iñochu k'enati madireši yichilalu? (Lemisalē 'kesenyo isike aribi' weyimi 'yesaminitu k'enati')"
        },
        "expiry_date_noted": {
            "english": "Expiry date noted as: {input}. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')",
            "amharic": "የማበቃበት ቀን እንደ {input} ተያዘ። {product_name}ን በየትኞቹ ቀናት ማድረስ ይችላሉ? (ለምሳሌ 'ከሰኞ እስከ አርብ' ወይም 'የሳምንቱ ቀናት')",
            "phonetic_amharic": "Yemabek'abeti k'eni inde {input} teyaze. {product_name}ni beyet'iñochu k'enati madireši yichilalu? (Lemisalē 'kesenyo isike aribi' weyimi 'yesaminitu k'enati')"
        },
        "no_expiry_noted": {
            "english": "Noted - {product_name} has no expiry date. What days can you deliver {product_name}? (e.g., 'Monday to Friday' or 'weekdays')",
            "amharic": "ተያዘ - {product_name} የማበቃበት ቀን የለውም። {product_name}ን በየትኞቹ ቀናት ማድረስ ይችላሉ? (ለምሳሌ 'ከሰኞ እስከ አርብ' ወይም 'የሳምንቱ ቀናት')",
            "phonetic_amharic": "Teyaze - {product_name} yemabek'abeti k'eni yelewimi. {product_name}ni beyet'iñochu k'enati madireši yichilalu? (Lemisalē 'kesenyo isike aribi' weyimi 'yesaminitu k'enati')"
        },
        "price_set": {
            "english": "Great! I'll set the price at {unit_price} ETB per kg. When does this {product_name} expire? (You can say 'no expiry' if it doesn't expire)",
            "amharic": "በጣም ጥሩ! ዋጋን በ {unit_price} ብር ለኪሎ እሰብራለሁ። ይህ {product_name} መቼ ነው የሚያልቀው? (የሚያበቃበት ቀን ከሌለው 'አያልቅም' ማለት ይችላሉ)",
            "phonetic_amharic": "Betam t'iru! Wagawini be {unit_price} birr le kīlo isebiralahu. Yih {product_name} mechē newi yemīyalk'ewi? (Yemīyabek'abeti k'eni kelēlewi 'ayalik'imi' maleti yichilalu)"
        },
        "need_product_quantity": {
            "english": "I need the product name and quantity first.",
            "amharic": "መጀመሪያ የምርት ስም እና ብዛት ያስፈልጉኛል።",
            "phonetic_amharic": "Mejemeriya yemirit sim ina bizati yasfeligugnal."
        },
        "what_product_pricing": {
            "english": "Which product do you need pricing insights for?",
            "amharic": "ለምን ምርት የዋጋ ምክር ያስፈልግዎታል?",
            "phonetic_amharic": "Lemin miriti yewaga mikir yasfeligotal?"
        },
        "no_competitor_data": {
            "english": "No competitor pricing data available for {product_name}.",
            "amharic": "ለ{product_name} የተፈጥሮአል የዋጋ መረጃ አይገኝም።",
            "phonetic_amharic": "Le{product_name} yetefet'iro'al yewaga mereja ayigenyim."
        },
        "competitor_price": {
            "english": "Average competitor price for {product_name}: {avg_price} ETB per kg.",
            "amharic": "ለ{product_name} የተፈጥሮአል አማካይ ዋጋ፡ {avg_price} ብር ለኪሎ።",
            "phonetic_amharic": "Le{product_name} yetefet'iro'al āmakayi waga: {avg_price} birr le kīlo."
        },
        "pricing_error": {
            "english": "I couldn't get pricing insights for {product_name}.",
            "amharic": "ለ{product_name} የዋጋ ምክር ማግኘት አልቻልኩም።",
            "phonetic_amharic": "Le{product_name} yewaga mikir maginet alchalkum."
        },
        "what_product_image": {
            "english": "Which product would you like an image for?",
            "amharic": "ለምን ምርት ምስል ያስፈልግዎታል?",
            "phonetic_amharic": "Lemin miriti misil yasfeligotal?"
        },
        "image_generated": {
            "english": "Image generated for {product_name}: {result}",
            "amharic": "ለ{product_name} ምስል ተፈጥሯል፡ {result}",
            "phonetic_amharic": "Le{product_name} misil tefet'irwal: {result}"
        },
        "image_error": {
            "english": "I couldn't generate an image for {product_name}.",
            "amharic": "ለ{product_name} ምስል መፍጠር አልቻልኩም።",
            "phonetic_amharic": "Le{product_name} misil mefṭer alchalkum."
        },
        "no_inventory": {
            "english": "You don't have any products in inventory yet.",
            "amharic": "እስካሁን ምንም ምርቶች ክምችት ውስጥ የለዎትም።",
            "phonetic_amharic": "Iska hun minim miritochi kimichit wisit'i yelewotimi."
        },
        "inventory_header": {
            "english": "📦 **Your Current Inventory:**",
            "amharic": "📦 **የእርስዎ አሁኑ ክምችት፡**",
            "phonetic_amharic": "📦 **Yerswo ahunu kimichiti:**"
        },
        "inventory_item": {
            "english": "{status_emoji} **{name}** • Quantity: {quantity} {unit} • Price: {price} ETB/{unit} • Delivery: {delivery_days}{expiry_info}",
            "amharic": "{status_emoji} **{name}** • ብዛት፡ {quantity} {unit} • ዋጋ፡ {price} ብር/{unit} • ማድረሻ፡ {delivery_days}{expiry_info}",
            "phonetic_amharic": "{status_emoji} **{name}** • Bizati: {quantity} {unit} • Waga: {price} birr/{unit} • Madireša: {delivery_days}{expiry_info}"
        },
        "inventory_error": {
            "english": "I couldn't check your inventory.",
            "amharic": "ክምችትዎን ማረጋገጥ አልቻልኩም።",
            "phonetic_amharic": "Kimichitiwon maregageti alchalkum."
        },
        "expiring_products": {
            "english": "Checking products expiring within {time_horizon}.",
            "amharic": "በ{time_horizon} ውስጥ የሚያልቁ ምርቶችን በማረጋገጥ ላይ።",
            "phonetic_amharic": "Be{time_horizon} wisit'i yemīyalk'u miritochini bemaregageti layi."
        },
        "what_flash_sale_accept": {
            "english": "Which product flash sale would you like to accept?",
            "amharic": "የትኛውን ምርት የፍላሽ ሽያጭ መቀበል ይፈልጋሉ?",
            "phonetic_amharic": "Yet'iñawini miriti yefilashi shiyachi mek'ibel yifeligalu?"
        },
        "flash_sale_accepted": {
            "english": "Flash sale accepted for {product_name}.",
            "amharic": "ለ{product_name} የፍላሽ ሽያጭ ተቀበለ።",
            "phonetic_amharic": "Le{product_name} yefilashi shiyachi tekebele."
        },
        "what_flash_sale_decline": {
            "english": "Which product flash sale would you like to decline?",
            "amharic": "የትኛውን ምርት የፍላሽ ሽያጭ መራቅ ይፈልጋሉ?",
            "phonetic_amharic": "Yet'iñawini miriti yefilashi shiyachi merak'i yifeligalu?"
        },
        "flash_sale_declined": {
            "english": "Flash sale declined for {product_name}.",
            "amharic": "ለ{product_name} የፍላሽ ሽያጭ ተራቁ።",
            "phonetic_amharic": "Le{product_name} yefilashi shiyachi terak'u."
        },
        "delivery_schedule": {
            "english": "Here's your delivery schedule for {date_range}.",
            "amharic": "የእርስዎ የማድረሻ መርሃ ግብር ለ{date_range} እነሆ።",
            "phonetic_amharic": "Yerswo yemadireša meriha gibiri le{date_range} ineho."
        },
        "login_supplier": {
            "english": "Please log in as a supplier first.",
            "amharic": "እባክዎ መጀመሪያ እንደ አቅራቢ ይግቡ።",
            "phonetic_amharic": "Ibakwo mejemeriya inide āk'rabī yigibu."
        },
        "what_date_deliveries": {
            "english": "Which date would you like to check deliveries for?",
            "amharic": "ለትኛ ቀን ማድረሻዎችን ማረጋገጥ ይፈልጋሉ?",
            "phonetic_amharic": "Let'iña k'eni madirešawochini maregageti yifeligalu?"
        },
        "no_deliveries_date": {
            "english": "You have no deliveries scheduled for {date}.",
            "amharic": "ለ{date} ምንም የተቀጠረ ማድረሻ የለዎትም።",
            "phonetic_amharic": "Le{date} minim yetek'et'ere madireša yelewotimi."
        },
        "deliveries_date": {
            "english": "Your deliveries for {date}:",
            "amharic": "የእርስዎ ማድረሻዎች ለ{date}፡",
            "phonetic_amharic": "Yerswo madirešawochi le{date}:"
        },
        "deliveries_error": {
            "english": "I couldn't check your deliveries for {date}. Please try again.",
            "amharic": "ለ{date} ማድረሻዎችዎን ማረጋገጥ አልቻልኩም። እባክዎ እንደገና ይሞክሩ።",
            "phonetic_amharic": "Le{date} madirešawochiwoni maregageti alchalkum. Ibakwo inidegena yimokiru."
        },
        "no_pending_orders": {
            "english": "You have no pending orders at this time.",
            "amharic": "እስካሁን ምንም ያልተሰማሩ ትእዛዞች የለዎትም።",
            "phonetic_amharic": "Iska hun minim yalitesemaru ti'izazochi yelewotimi."
        },
        "order_not_found": {
            "english": "I couldn't find an order with reference '{order_ref}' in your pending orders.",
            "amharic": "በያልተሰማሩ ትእዛዞችዎ ውስጥ በ'{order_ref}' ማጣቀሻ የሆነ ትእዛዝ ማግኘት አልቻልኩም።",
            "phonetic_amharic": "Be yalitesemaru ti'izazochiwo wisit'i be'{order_ref}' mat'ak'isha yehone ti'izaz maginet alchalkum."
        },
        "what_order_accept": {
            "english": "Which order would you like to accept?",
            "amharic": "የትኛውን ትእዛዝ መቀበል ይፈልጋሉ?",
            "phonetic_amharic": "Yet'iñawini ti'izazi mek'ibel yifeligalu?"
        },
        "order_accepted": {
            "english": "Order {order_ref} accepted.",
            "amharic": "ትእዛዝ {order_ref} ተቀበለ።",
            "phonetic_amharic": "Ti'izaz {order_ref} tekebele."
        },
        "what_product_quantity_update": {
            "english": "What product and how much quantity do you want to add?",
            "amharic": "ምን ምርት እና ምን ያህል ብዛት መጨመር ይፈልጋሉ?",
            "phonetic_amharic": "Min miriti ina min yahili bizati mech'emeri yifeligalu?"
        },
        "product_not_found": {
            "english": "I couldn't find {product_name} in your inventory. Please add it as a new product first.",
            "amharic": "በክምችትዎ ውስጥ {product_name} ማግኘት አልቻልኩም። እባክዎ መጀመሪያ እንደ አዲስ ምርት ያክሉት።",
            "phonetic_amharic": "Be kimichitiwo wisit'i {product_name} maginet alchalkum. Ibakwo mejemeriya inide addis miriti yakiluti."
        },
        "not_in_inventory": {
            "english": "You don't have {product_name} in your inventory yet. Please add it as a new product first.",
            "amharic": "እስካሁን በክምችትዎ ውስጥ {product_name} የለዎትም። እባክዎ መጀመሪያ እንደ አዲስ ምርት ያክሉት።",
            "phonetic_amharic": "Iska hun be kimichitiwo wisit'i {product_name} yelewotimi. Ibakwo mejemeriya inide addis miriti yakiluti."
        },
        "inventory_updated": {
            "english": "Added {quantity} kg to your existing {product_name} inventory. Total quantity now: {new_quantity} kg at {current_price} ETB per kg, deliverable {delivery_days}.",
            "amharic": "ወደ ክምችትዎ ያለው {product_name} {quantity} ኪግ ተጨምሯል። አሁን ጠቅላላ ብዛት፡ {new_quantity} ኪግ በ{current_price} ብር ለኪሎ፣ ማድረሻ {delivery_days}።",
            "phonetic_amharic": "Wede kimichitiwo yalewi {product_name} {quantity} kig techemirwal. Ahun timirali bizati: {new_quantity} kig be{current_price} birr le kīlo, madireša {delivery_days}."
        },
        "product_removed": {
            "english": "Removed {product_name} from your inventory.",
            "amharic": "ከክምችትዎ {product_name} ተወገደ።",
            "phonetic_amharic": "Ke kimichitiwo {product_name} tewegeḍe."
        },
        "inventory_update_error": {
            "english": "I couldn't update your {product_name} inventory. Please try again.",
            "amharic": "የ{product_name} ክምችትዎን ማሻሻል አልቻልኩም። እባክዎ እንደገና ይሞክሩ።",
            "phonetic_amharic": "Ye{product_name} kimichitiwon mashashali alchalkum. Ibakwo inidegena yimokiru."
        },

        # General errors
        "error_generic": {
            "english": "I encountered an issue processing your request. Please try again.",
            "amharic": "ጥያቄዎን በማስተናገድ ላይ ችግር አጋጥሞኛል። እባክዎ እንደገና ይሞክሩ።",
            "phonetic_amharic": "T'iyak'ēwoni bemasitenagedi layi chigiri agat'imoñali. Ibakwo inidegena yimokiru."
        },
        "error_unknown": {
            "english": "I'm here to help with your fresh produce needs. What would you like to do?",
            "amharic": "ለአዲስ ምርት ፍላጎቶችዎ ለመርዳት እዚህ ነኝ። ምን ማድረግ ይፈልጋሉ?",
            "phonetic_amharic": "Le'ādis miriti filagotochiwo lemeridati izīhi neñi. Mini madiregi yifeligalu?"
        },
        "error_supplier": {
            "english": "I'm here to help you manage your inventory and sales. What would you like to do?",
            "amharic": "ክምችትዎን እና ሽያጭዎን እንዲያስተዳድሩ ለመርዳት እዚህ ነኝ። ምን ማድረግ ይፈልጋሉ?",
            "phonetic_amharic": "Kimichitiwoni ina shiyach'iwoni inidīyasitedadiru lemeridati izīhi neñi. Mini madiregi yifeligalu?"
        },

        # Nutrition query responses
        "nutrition_query_missing_products": {
            "english": "Which two products would you like to compare nutritionally?",
            "amharic": "በንጥረ ነገር ይዘታቸው የትኞቹን ሁለት ምርቶች ማወዳደር ይፈልጋሉ?",
            "phonetic_amharic": "Benit'ire neger yizetachew yet'iñochun hulet miritochi mawedader yifeligalu?"
        },
        "nutrition_no_data": {
            "english": "I couldn't find nutritional information comparing {product_a} and {product_b}.",
            "amharic": "{product_a} እና {product_b}ን የሚያወዳድር የአመጋገብ መረጃ ማግኘት አልቻልኩም።",
            "phonetic_amharic": "{product_a} ina {product_b}n yemīyawedadiri ye'āmigabi mereja maginet alchalkum."
        },
        "nutrition_error": {
            "english": "I encountered an error while getting nutritional information for {product_a} and {product_b}.",
            "amharic": "ለ{product_a} እና {product_b} የአመጋገብ መረጃ በማምጣት ላይ ሳለ ስህተት አጋጥሞኛል።",
            "phonetic_amharic": "Le{product_a} ina {product_b} ye'āmigabi mereja bemamit'ati layi sale sihiteti agat'imoñali."
        }
    }