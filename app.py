import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="🍽️ Personalized Recipe Recommender",
    page_icon="🍲",
    layout="centered"
)

st.title("🍽️ Personalized Recipe Recommender")
st.markdown("""
Get customized recipe recommendations based on your dietary preferences, health goals, and nutritional needs!
""")

# Sidebar for User Input
st.sidebar.header("Your Preferences")

# Basic Information
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=175)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
activity_level = st.sidebar.selectbox(
    "Activity Level",
    ["Low", "Medium", "High"]
)
goal = st.sidebar.selectbox(
    "Goal",
    ["Lose Weight", "Maintain Weight", "Gain Weight"]
)
diet_type = st.sidebar.selectbox(
    "Diet Type",
    ["Vegetarian", "Non-Vegetarian"]
)

# 4. Nutritional Preferences (Sliders)
st.sidebar.subheader("Nutritional Preferences")
fat_score = st.sidebar.slider("Fat Intake", 0, 2, 1, help="0=Low, 1=Medium, 2=High")
sugar_score = st.sidebar.slider("Sugar Intake", 0, 2, 1)
sodium_score = st.sidebar.slider("Sodium Intake", 0, 2, 1)
protein_score = st.sidebar.slider("Protein Intake", 0, 2, 1)

user_prefs = {
    "Height": height,
    "Weight": weight,
    "Age": age,
    "Gender": gender.lower(),
    "Activity_level": activity_level.lower(),
    "Goal": goal.lower().replace(" weight", ""),
    "Veg_NonVeg": 0 if diet_type == "Vegetarian" else 1,  # Changed from "Veg/Non-Veg"
    "FatScore": fat_score,
    "SugarScore": sugar_score,
    "SatFatScore": fat_score,
    "SodiumScore": sodium_score,
    "CholesterolScore": 1,
    "FiberScore": 1,
    "ProteinScore": protein_score,
    "CarbScore": 1
}

# Fetch Recommendations from FastAPI
if st.sidebar.button("Get Recommendations 🍳"):
    with st.spinner("Finding the best recipes for you..."):
        try:
            # Make API call to FastAPI backend
            API_URL = "http://127.0.0.1:8001/recommend"  # Replace with your endpoint
            response = requests.post(API_URL, json=user_prefs)
            
            if response.status_code == 200:
                recommendations = response.json()
                
                st.success("🎉 Here are your personalized recommendations!")
                
                # Display Recipes
                st.subheader("Recommended Recipes")
                for recipe in recommendations:
                    with st.expander(f"🍲 {recipe['Name']}"):
                        st.markdown(f"**Calories:** {recipe['Calories']:.0f} kcal")
                        st.markdown(f"**Protein:** {recipe['ProteinContent']:.1f}g")
                        st.markdown(f"**Carbs:** {recipe['CarbohydrateContent']:.1f}g")
                        st.markdown(f"**Fat:** {recipe['FatContent']:.1f}g")
                        st.markdown("---")
                        # st.markdown("**Ingredients:**")
                        # st.write(recipe["RecipeIngredientParts"].replace(",", "\n- "))
                        # st.markdown("**Instructions:**")
                        # st.write(recipe["Recipe_Instructions"])
            else:
                st.error(f"⚠️ API Error: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"🚨 Failed to connect to API. Is it running? Error: {e}")

# Run the App
if __name__ == "__main__":
    st.write("Configure your preferences on the left and click **Get Recommendations**!")
