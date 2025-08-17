# recommender.py

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
import uvicorn
import streamlit
import requests
import fastapi
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer



def create_nutrition_scores(df):

    # FDA nutritional guidelines per serving (in grams)
    FDA_GUIDELINES = {
    'Fat': {'low': 3, 'high': 20},          # Total Fat
    'SatFat': {'low': 1, 'high': 5},        # Saturated Fat
    'Sugar': {'low': 5, 'high': 15},        # Total Sugars
    'Sodium': {'low': 140, 'high': 400},    # Sodium (mg)
    'Cholesterol': {'low': 20, 'high': 60}, # Cholesterol (mg)
    'Fiber': {'low': 1, 'high': 5},         # Dietary Fiber
    'Protein': {'low': 5, 'high': 20},      # Protein
    'Carbs': {'low': 5, 'high': 20}         # Total Carbohydrates
    }

    """
    Creates nutritional score columns (-1, 0, 1) based on FDA guidelines
    -1 = Low, 0 = Medium, 1 = High
    """
    # Fat Score
    df['FatScore'] = np.select(
        [
            df['FatContent'] <= FDA_GUIDELINES['Fat']['low'],
            df['FatContent'] >= FDA_GUIDELINES['Fat']['high']
        ],
        [0, 2],
        default=1
    )

    # Sugar Score
    df['SugarScore'] = np.select(
        [
            df['SugarContent'] <= FDA_GUIDELINES['Sugar']['low'],
            df['SugarContent'] >= FDA_GUIDELINES['Sugar']['high']
        ],
        [0, 2],
        default=1
    )

    # Saturated Fat Score
    df['SatFatScore'] = np.select(
        [
            df['SaturatedFatContent'] <= FDA_GUIDELINES['SatFat']['low'],
            df['SaturatedFatContent'] >= FDA_GUIDELINES['SatFat']['high']
        ],
        [0, 2],
        default=1
    )

    # Sodium Score (convert mg to grams if needed)
    df['SodiumScore'] = np.select(
        [
            df['SodiumContent'] <= FDA_GUIDELINES['Sodium']['low'],
            df['SodiumContent'] >= FDA_GUIDELINES['Sodium']['high']
        ],
        [0, 2],
        default=1
    )

    # Cholesterol Score
    df['CholesterolScore'] = np.select(
        [
            df['CholesterolContent'] <= FDA_GUIDELINES['Cholesterol']['low'],
            df['CholesterolContent'] >= FDA_GUIDELINES['Cholesterol']['high']
        ],
        [0, 2],
        default=1
    )

    # Fiber Score (positive scoring - more fiber is better)
    df['FiberScore'] = np.select(
        [
            df['FiberContent'] >= FDA_GUIDELINES['Fiber']['high'],
            df['FiberContent'] <= FDA_GUIDELINES['Fiber']['low']
        ],
        [0, 2],
        default=1
    )

    # Protein Score
    df['ProteinScore'] = np.select(
        [
            df['ProteinContent'] >= FDA_GUIDELINES['Protein']['high'],
            df['ProteinContent'] <= FDA_GUIDELINES['Protein']['low']
        ],
        [0, 2],
        default=1
    )

    # Carbs Score
    df['CarbScore'] = np.select(
        [
            df['CarbohydrateContent'] <= FDA_GUIDELINES['Carbs']['low'],
            df['CarbohydrateContent'] >= FDA_GUIDELINES['Carbs']['high']
        ],
        [0, 2],
        default=1
    )

    return df

def classify_diet(ingredients):

    non_veg_keywords = [
    # Meats
    "chicken", "poultry", "hen", "rooster",
    "beef", "steak", "veal",
    "pork", "bacon", "ham", "sausage", "prosciutto", "pepperoni",
    "lamb", "mutton", "goat",
    "duck", "goose", "quail", "turkey",
    "venison", "bison",  "game meat",

    # Fish & Seafood
    "fish", "salmon", "tuna", "cod", "sardine", "mackerel",
    "trout", "halibut", "anchovy", "bass", "tilapia",
    "shrimp", "prawn", "crab", "lobster", "crayfish",
    "oyster", "clam", "mussel", "scallop", "squid", "octopus",

    # Eggs & Dairy
    "egg", "eggs", "egg white", "egg yolk", "mayonnaise",
    "gelatin", "rennet", "lard", "tallow", "suet",

    # Meat-based broths/sauces
    "chicken broth", "beef broth", "fish sauce", "oyster sauce",
    "worcestershire sauce", "dashi"  # (often contains fish)
    ]

    ingredients = ingredients.lower()
    if any(keyword in ingredients for keyword in non_veg_keywords):
        return 1
    else:
        return 0

def calculate_daily_calories(height, weight, age, gender, activity_level, goal):
    """
    Calculate daily calorie needs using Harris-Benedict equation
    """
    if gender.lower() == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    activity_multiplier = {
        'low': 1.2,
        'medium': 1.55,
        'high': 1.725
    }

    tdee = bmr * activity_multiplier[activity_level.lower()]

    if goal.lower() == 'lose':
        return tdee - 500  # 500 calorie deficit for weight loss
    elif goal.lower() == 'gain':
        return tdee + 500   # 500 calorie surplus for weight gain
    else:
        return tdee         # maintenance


def prepare_input_vector(user_prefs, calories,df):
    """
    Create an input vector matching our feature space
    """
    # Get median values for non-specified nutritional content
    median_values = df.iloc[:, 6:15].median()

    # Create input vector
    input_vector = [
        calories,
        median_values['FatContent'],      # Will be adjusted by score
        median_values['SaturatedFatContent'],
        median_values['CholesterolContent'],
        median_values['SodiumContent'],
        median_values['CarbohydrateContent'],
        median_values['FiberContent'],
        median_values['SugarContent'],
        median_values['ProteinContent'],
        user_prefs['Veg/Non-Veg'],
        user_prefs['FatScore'],
        user_prefs['SugarScore'],
        user_prefs['SatFatScore'],
        user_prefs['SodiumScore'],
        user_prefs['CholesterolScore'],
        user_prefs['FiberScore'],
        user_prefs['ProteinScore'],
        user_prefs['CarbScore']
    ]

    return np.array([input_vector])

def recommend_recipes(user_prefs, df, pipeline, n_recommendations=5):
    """
    Get personalized recipe recommendations
    """
    # Get user preferences (using dictionary access)
    calories = calculate_daily_calories(
        user_prefs['Height'], 
        user_prefs['Weight'], 
        user_prefs['Age'], 
        user_prefs['Gender'], 
        user_prefs['Activity_level'], 
        user_prefs['Goal']
    )
        # Prepare input vector
    input_vector = prepare_input_vector(user_prefs, calories, df)

    # Get recommendations
    indices = pipeline.transform(input_vector)[0]

    # Filter by diet type
    recommendations = df.iloc[indices]
    recommendations = recommendations[recommendations['Veg/Non-Veg'] == user_prefs['Veg/Non-Veg']]

    # Return top recommendations
    return recommendations.head(n_recommendations)

def preprocess():
   
   # setting up data
    df = pd.read_csv('recipes.csv')
    df.drop(['RecipeId','AuthorName','Images','RecipeYield','ReviewCount','AuthorId','DatePublished','AggregatedRating','PrepTime','CookTime','TotalTime'],axis=1,inplace=True)
    df = df.dropna(how='any')

   # data preparation  
    # df['Calories']=df['Calories']/df['RecipeServings']
    df['Calories']=df['Calories']
    df['FatContent']=df['FatContent']
    df['SaturatedFatContent']=df['SaturatedFatContent']
    df['CarbohydrateContent']=df['CarbohydrateContent']
    df['CholesterolContent']=df['CholesterolContent']
    df['SodiumContent']=df['SodiumContent']
    df['FiberContent']=df['FiberContent']
    df['SugarContent']=df['SugarContent']
    df['ProteinContent']=df['ProteinContent']
    # df['FatContent']=df['FatContent']/df['RecipeServings']
    # df['SaturatedFatContent']=df['SaturatedFatContent']/df['RecipeServings']
    # df['CarbohydrateContent']=df['CarbohydrateContent']/df['RecipeServings']
    # df['CholesterolContent']=df['CholesterolContent']/df['RecipeServings']
    # df['SodiumContent']=df['SodiumContent']/df['RecipeServings']
    # df['FiberContent']=df['FiberContent']/df['RecipeServings']
    # df['SugarContent']=df['SugarContent']/df['RecipeServings']
    # df['ProteinContent']=df['ProteinContent']/df['RecipeServings']  

    #feature engineering

    df = create_nutrition_scores(df)
    
    # Classify Diet Type

    df['Veg/Non-Veg'] = df['RecipeIngredientParts'].apply(classify_diet)

    df['Recipe_Instructions']=df['RecipeInstructions']*1

    df.drop(['RecipeInstructions'],axis=1,inplace=True)
    df.drop(['RecipeServings'], axis=1,inplace=True)
    return df.reset_index(drop=True)

def train(df):
    X=df
    X_train, X_test = train_test_split(X, test_size = 0.2, random_state = 0)
    scaler=StandardScaler()
    prep_data=scaler.fit_transform(X_train.iloc[:,6:24].to_numpy())

    #Training the model
    neigh = NearestNeighbors(metric='cosine',algorithm='brute')
    neigh.fit(prep_data)

    transformer = FunctionTransformer(neigh.kneighbors,kw_args={'return_distance':False})
    pipeline=Pipeline([('std_scaler',scaler),('NN',transformer)])
    
    params={'n_neighbors':10,'return_distance':False}
    pipeline.get_params()
    pipeline.set_params(NN__kw_args=params)
    return pipeline














