"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","About Us","Exploratory Data Analysis","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[149:152])
        movie_2 = st.selectbox('Second Option',title_list[250:352])
        movie_3 = st.selectbox('Third Option',title_list[200:212])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    #if page_selection == "Solution Overview":
        #st.title("Solution Overview")
        #st.write("Describe your winning approach on this page")
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        #st.write("Describe your winning approach on this page")
        #st.title("Solution Overview")
        # st.write("Describe your winning approach on this page")
        if st.button("Click to compare Models"):
        # Call your model prediction functions here and store the results
        # You can use the user inputs from the input widgets as inputs to the models
            with st.expander("Model Definitions"):
                # Display the results side by side
                st.subheader("Non_Negative Matrix Factorization Model:")
                st.write("The Non-Negative Matrix Factorization (NMF) algorithm leverages principles from multivariate analysis \
                        and linear algebra. Its purpose is to break down a given data matrix, represented as M, into the product of two matrices with lower ranks, \
                        namely W and H. The sub-matrix W represents the NMF basis, while the sub-matrix H contains the corresponding coefficients.")
                
                st.subheader("Single Value Decomposition Model:")
                st.write("Single Value Decomposition (SVD) is a fundamental matrix factorization technique used in linear algebra and numerical analysis. \
                        The beauty of SVD lies in its ability to reduce the dimensionality of the original matrix while preserving its essential information. \
                        By truncating the number of singular values and their corresponding vectors, we can approximate the original matrix using a lower-rank approximation.\
                        It can be used to factorize a user-item interaction matrix to identify latent factors (features) that contribute to the preferences of users for different items. \
                        This technique is known as matrix factorization and has been successfully applied in recommender systems to make personalized recommendations.")

                st.subheader("Co-Clustering Model:")
                st.write("Co-Clustering, also known as biclustering or co-clustering, is a machine learning technique used to simultaneously cluster both rows and columns of a data matrix. \
                        It is particularly useful when dealing with data that exhibits natural grouping patterns in both dimensions. The goal of the Co-Clustering algorithm is to find an optimal \
                        partitioning of the rows and columns that maximizes a clustering criterion, such as minimizing the sum of squared differences between elements within each cluster.")

            with st.expander("Model Performance"):
                st.write("RMSE is a popular evaluation metric in various machine learning tasks, particularly in regression problems, where the goal is to predict a continuous numerical output. \
                         It provides a measure of the model's average prediction error, with a lower RMSE indicating better predictive accuracy.")
                rmse_scores = [0.78, 0.81, 0.89]
                trained_models = ['Single Value Decomposition', 'Single Value Decomposition', 'Co-Clustering']

                model_performance = pd.DataFrame({'Model': trained_models, 'RMSE': rmse_scores})

                # Display the model performance table
                st.subheader("Model Performance:")
                st.table(model_performance)

                # Highlight the best model based on RMSE score
                best_model = model_performance.loc[model_performance['RMSE'].idxmin(), 'Model']
                st.subheader(f"The best model is {best_model} with RMSE: {model_performance['RMSE'].min()}")
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "About Us":
        st.write("### SmartByte Inc")
        
        # You can read a markdown file from supporting resources folder
        #if st.checkbox("Introduction"):
        st.subheader("Where Data Meets Intelligence")
        st.write("""Welcome to SmartByte Inc. – Where Data Meets Intelligence!

At SmartByte Inc., we are passionate about the transformative power of data science and its ability to revolutionize businesses across all industries. Our mission is to provide cutting-edge data solutions that empower companies to make data-driven decisions, gain valuable insights, and achieve unparalleled success in today's data-driven world.

Who We Are:
SmartByte Inc. is a leading data science company, comprised of a team of highly skilled data scientists, analysts, engineers, and industry experts. Our diverse and talented team is united by a common goal: to harness the potential of data and turn it into actionable intelligence for our clients.

What We Do:
We offer a comprehensive suite of data science services, tailored to meet the unique needs of businesses, large or small. From startups looking to gain a competitive edge to established enterprises aiming to optimize their operations, our solutions cater to all.

1. Data Analysis and Insights:
Our data experts excel at collecting, cleaning, and analyzing vast amounts of data, uncovering hidden patterns, and extracting valuable insights. We help businesses understand their data, identify key trends, and make informed decisions that lead to increased efficiency and profitability.

2. Machine Learning and AI Solutions:
Leveraging the power of machine learning and artificial intelligence, we develop custom algorithms and models to solve complex business challenges. Whether it's predictive analytics, recommendation systems, or natural language processing, our AI solutions pave the way for intelligent automation and enhanced customer experiences.

3. Data Visualization:
We understand that making data understandable and accessible is crucial for successful decision-making. Our data visualization experts create compelling and interactive dashboards that bring data to life, enabling our clients to grasp complex information effortlessly.

4. Big Data Infrastructure:
In the era of big data, having the right infrastructure is vital. We help businesses build robust and scalable data infrastructure, ensuring they can handle large volumes of data while maintaining data security and compliance.

Why Choose SmartByte Inc.:
- Expertise: Our team comprises industry-leading data scientists and professionals with a wealth of experience across diverse domains.
- Innovation: We stay ahead of the curve by constantly exploring emerging technologies and trends in the data science landscape.
- Customization: Every business is unique, and we tailor our solutions to address specific challenges and opportunities faced by our clients.
- Client-Centric Approach: At SmartByte Inc., our clients' success is our top priority. We collaborate closely with our clients, ensuring their needs are met and expectations exceeded.

Our Commitment:
Data integrity, privacy, and security are at the core of everything we do. We adhere to the highest standards of data governance, ensuring our clients' data is handled responsibly and ethically.

Join us on this exciting journey as we unlock the true potential of data science for your business. Together, we'll turn data into actionable intelligence, propelling your organization towards a brighter and more successful future.

Contact us today to discover how SmartByte Inc. can transform your data into your most valuable asset. Let's embark on this data-driven adventure together!""")
        #st.write("""With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.""")
        #st.write("""Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.""")

        #if st.checkbox("Problem Statement"):
        #st.subheader("Problem Statement of the Unsupervised Learning Predict")
        #st.write("Build recommender systems to recommend a movie")

        #if st.checkbox("Data"):
        #st.subheader("Data Overview")
        #st.write("""This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!""")

        #st.write("""For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.""")

        #st.write("""### Source:""") 
        #st.write("""The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB""")


        #st.write("""### Supplied Files:
        #genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here

        #genome_tags.csv - user assigned tags for genome-related scores

        #imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.

        #links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.

        #sample_submission.csv - Sample of the submission format for the hackathon.

        #tags.csv - User assigned for the movies within the dataset.

        #test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.

        #train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.""")    
    if page_selection == "Exploratory Data Analysis":
        st.title('Exploratory Data Analysis')

        if st.checkbox("ratings"):
            st.subheader("Movie ratings")
            #st.image('resources/imgs/rating.PNG',use_column_width=True)

        # if st.checkbox("correlation"):
        #     st.subheader("Correlation between features")
        #     st.image('resources/imgs/correlation.png',use_column_width=True)
        
        if st.checkbox("genre wordcloud"):
            st.subheader("Top Genres")
            #st.image('resources/imgs/genre_wordcloud.png',use_column_width=True)
        
        if st.checkbox("genres"):
            st.subheader("Top Genres")
            #st.image('resources/imgs/top_genres.PNG',use_column_width=True)
        
        # if st.checkbox("movies released per year"):
        #     st.subheader("Movies released per year")
        #     st.image('resources/imgs/release_year.png',use_column_width=True)

        if st.checkbox("tags"):
            st.subheader("Top tags")
            #st.image('resources/imgs/top_tags.PNG',use_column_width=True)

        if st.checkbox("cast"):
            st.subheader("Popular cast")
            #st.image('resources/imgs/cast.PNG',use_column_width=True)


if __name__ == '__main__':
    main()
