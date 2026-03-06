import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_complex_user_data(num_users=1000):
    user_data = []
    
    # Categorical Options
    locations = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune', 'Chennai', 'Kolkata']
    genders = ['Male', 'Female', 'Non-Binary']
    occupations = ['Student', 'Software Engineer', 'Doctor', 'Artist', 'Teacher', 'Entrepreneur']
    devices = ['iOS', 'Android', 'Windows', 'MacOS']
    tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    interests = ['Tech', 'Fashion', 'Fitness', 'Home Decor', 'Gaming', 'Books', 'Beauty']

    for i in range(num_users):
        u_id = f"USR_{2000 + i}"
        
        # --- 1. Demographics (10 Cols) ---
        age = random.randint(18, 65)
        gender = random.choice(genders)
        loc = random.choice(locations)
        tier = random.choice(tiers)
        occ = random.choice(occupations)
        device = random.choice(devices)
        income_bracket = random.choice(['Low', 'Medium', 'High', 'Ultra-High'])
        marital_status = random.choice(['Single', 'Married'])
        has_kids = random.choice([0, 1])
        preferred_lang = random.choice(['English', 'Hindi', 'Regional'])

        # --- 2. Behavioral Aggregates - Last 30 Days (15 Cols) ---
        total_clicks = random.randint(10, 500)
        total_likes = random.randint(0, 100)
        total_shares = random.randint(0, 20)
        avg_dwell_time = round(random.uniform(5.0, 300.0), 2) # in seconds
        bounce_rate = round(random.uniform(0.1, 0.8), 2)
        cart_add_count = random.randint(0, 30)
        search_freq = random.randint(5, 100)
        weekend_activity_ratio = round(random.uniform(0.1, 0.9), 2)
        night_owl_score = round(random.uniform(0.0, 1.0), 2) # Activity between 11PM-4AM
        morning_bird_score = round(random.uniform(0.0, 1.0), 2)
        app_open_freq = random.randint(1, 15) # Daily
        notification_click_rate = round(random.uniform(0.0, 0.5), 2)
        last_active_days_ago = random.randint(0, 30)
        session_depth = random.randint(1, 20) # Pages per session
        churn_probability = round(random.uniform(0.0, 1.0), 2)

        # --- 3. Financial & Purchase Context (10 Cols) ---
        avg_ticket_size = random.randint(200, 25000)
        max_spend_single_item = random.randint(avg_ticket_size, 50000)
        total_lifetime_spend = random.randint(1000, 500000)
        discount_affinity = round(random.uniform(0.1, 0.9), 2) # Likes sales?
        premium_brand_affinity = round(random.uniform(0.0, 1.0), 2)
        return_rate = round(random.uniform(0.0, 0.3), 2)
        cod_preference = random.choice([0, 1]) # Cash on delivery
        wallet_user = random.choice([0, 1])
        loyalty_points = random.randint(0, 5000)
        membership_type = random.choice(['Free', 'Silver', 'Gold', 'Platinum'])

        # --- 4. Category Affinities (10 Cols) ---
        # How much they like each category (0.0 to 1.0)
        aff_electronics = round(random.uniform(0, 1), 2)
        aff_fashion = round(random.uniform(0, 1), 2)
        aff_fitness = round(random.uniform(0, 1), 2)
        aff_home = round(random.uniform(0, 1), 2)
        aff_beauty = round(random.uniform(0, 1), 2)
        aff_groceries = round(random.uniform(0, 1), 2)
        aff_books = round(random.uniform(0, 1), 2)
        aff_gaming = round(random.uniform(0, 1), 2)
        aff_lifestyle = round(random.uniform(0, 1), 2)
        aff_travel = round(random.uniform(0, 1), 2)

        # --- 5. Semantic Bio & Search (5 Cols) ---
        primary_interest = random.choice(interests)
        bio = f"{occ} from {loc}, loves {primary_interest}. Frequent {device} user."
        most_searched_brand = random.choice(['Apple', 'Nike', 'Samsung', 'Zara', 'LG'])
        user_intent_score = round(random.uniform(0, 1), 2) # 1 = Buying, 0 = Browsing
        last_search_query = f"Best {random.choice(interests)} products"

        # Combine all 50 columns
        user_data.append([
            u_id, age, gender, loc, tier, occ, device, income_bracket, marital_status, has_kids, preferred_lang,
            total_clicks, total_likes, total_shares, avg_dwell_time, bounce_rate, cart_add_count, search_freq,
            weekend_activity_ratio, night_owl_score, morning_bird_score, app_open_freq, notification_click_rate,
            last_active_days_ago, session_depth, churn_probability,
            avg_ticket_size, max_spend_single_item, total_lifetime_spend, discount_affinity, premium_brand_affinity,
            return_rate, cod_preference, wallet_user, loyalty_points, membership_type,
            aff_electronics, aff_fashion, aff_fitness, aff_home, aff_beauty, aff_groceries, aff_books, aff_gaming, aff_lifestyle, aff_travel,
            bio, most_searched_brand, user_intent_score, last_search_query
        ])

    columns = [
        'user_id', 'age', 'gender', 'location', 'city_tier', 'occupation', 'device_type', 'income_level', 'marital_status', 'has_kids', 'language',
        'total_clicks_30d', 'total_likes_30d', 'total_shares_30d', 'avg_dwell_time', 'bounce_rate', 'cart_add_30d', 'search_count_30d',
        'weekend_ratio', 'night_owl_score', 'morning_bird_score', 'app_open_daily', 'notif_click_rate',
        'last_active_days', 'avg_session_depth', 'churn_prob',
        'avg_ticket_size', 'max_item_spend', 'total_ltv', 'discount_affinity', 'premium_affinity',
        'return_rate', 'cod_user', 'wallet_user', 'loyalty_points', 'membership_level',
        'aff_electronics', 'aff_fashion', 'aff_fitness', 'aff_home', 'aff_beauty', 'aff_groceries', 'aff_books', 'aff_gaming', 'aff_lifestyle', 'aff_travel',
        'profile_bio', 'top_searched_brand', 'buying_intent_score', 'last_search_query'
    ]
    
    return pd.DataFrame(user_data, columns=columns)

# Run and save
df_users = generate_complex_user_data(1000)
df_users.to_csv('user_data_50_cols.csv', index=False)
print("Dataset Generated with 50 Columns!")