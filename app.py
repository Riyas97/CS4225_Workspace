import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import datetime as dt
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
from scipy.stats.stats import kendalltau

st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv("part-00000-59491de8-9173-41d4-b610-f61c71022b94-c000.csv")
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['date'] = df['date'].dt.date

st.title("Sentiment Analysis of Tweets about COVID-19")
st.sidebar.title("Lets' get started!")
st.markdown("This is a streamlit dashboard application that visualizes people's general sentiment towards COVID-19 in several countries/areas.")
st.sidebar.markdown(
    "Follow the instructions below to start exploring the visualizations!")


def run():

    pages = ["Visualize Single Country/Area",
             "Visualize Multiple Countries/Areas"]
    st.sidebar.subheader("First, choose a visualization type...")
    section = st.sidebar.radio('Visualization selected:', pages)
    countries = ['Singapore', 'United States', 'Italy', 'Germany', 'Norway', 'United Kingdom', 'Australia', 'Taiwan',
                 'Hong Kong', 'China', 'Brazil', 'France', 'New Zealand', 'South Korea', 'Japan',
                 'Vietnam', 'India', 'Canada', 'Saudi Arabia', 'Bahrain',
                 'South Africa', 'Egypt', 'Argentina']

    if section == "Visualize Single Country/Area":

        st.sidebar.subheader("Now, select a country to visualize...")
        country_selected = st.sidebar.selectbox(
            'Country Selected:', ['Singapore', 'United States', 'Italy', 'Germany', 'Norway', 'United Kingdom', 'Australia', 'Taiwan',
                                  'Hong Kong', 'China', 'Brazil', 'France', 'New Zealand', 'South Korea', 'Japan',
                                  'Vietnam', 'India', 'Canada', 'Saudi Arabia', 'Bahrain',
                                  'South Africa', 'Egypt', 'Argentina'])

        st.sidebar.subheader("Now, select a time frame...")
        start_date = st.sidebar.date_input(
            "Select a start date:",
            min_value=dt.date(2022, 1, 1),
            value=dt.date(2022, 1, 1),
            max_value=dt.date(2022, 2, 28))
        end_date = st.sidebar.date_input(
            "Select a end date:",
            min_value=dt.date(2022, 1, 2),
            value=dt.date(2022, 1, 2),
            max_value=dt.date(2022, 2, 28))

        if start_date < end_date:
            pass
        else:
            st.error('Error: End date must fall after start date.')

        #startdate = pd.to_datetime("2017-7-7").date()
        #enddate = pd.to_datetime("2017-7-10").date()

        if st.sidebar.button('Create Visualization'):

            df_filtered = df.loc[df['location'] == country_selected]
            df_filtered = df_filtered.loc[(df_filtered['date'] >= start_date) & (
                df_filtered['date'] <= end_date)]
            corr_df1 = df_filtered.corr(method='pearson')
            corr_df2 = df_filtered.corr(method='kendall')
            corr_df3 = df_filtered.corr(method='spearman')

            #fig = px.line(df_filtered, x="date", y=df.columns[2:8])

            fig1 = make_subplots(rows=1, cols=1)
            fig1.add_scatter(
                x=df_filtered["date"], y=df_filtered["stringency_index"], mode='lines', name="stringency_index")
            fig1.add_scatter(x=df_filtered["date"], y=df_filtered["%_of_positive_sentiments"],
                             mode='lines', name="%_of_positive_sentiments")
            fig1.add_scatter(x=df_filtered["date"], y=df_filtered["%_of_negative_sentiments"],
                             mode='lines', name="%_of_negative_sentiments")
            fig1.add_scatter(
                x=df_filtered["date"], y=df_filtered["%_of_mixed_sentiments"], mode='lines', name="%_of_mixed_sentiments")
            fig1.add_scatter(x=df_filtered["date"], y=df_filtered["%_of_neutral_sentiments"],
                             mode='lines', name="%_of_neutral_sentiments")

            fig1.update_layout(
                xaxis_title="Date",
                yaxis_title="Values",
                title="<b>Chart 1: Stringency Index vs Time & Percentage of Sentiments vs Time</b>",
                font=dict(
                    size=13
                )
            )

            st.plotly_chart(fig1)

            st.info(
                "🛈 Click on the respective variable legend (on the right of the chart) to select or deselect it.")

            metrics_meaning = st.expander('What does the metrics mean?')
            with metrics_meaning:
                st.markdown("stringency_index: Composite measure based on nine response indicators including school closures, workplace closures, and travel bans, rescaled to a value from 0 to 100 (100 = strictest). On the above chart, the value has been divided by 100 to make it easier to observe patterns in the chart")
                st.markdown(
                    "%_of_positive_sentiments: Percentage of tweets (belonging to a particular date) that registered a positive sentiment")
                st.markdown(
                    "%_of_negative_sentiments: Percentage of tweets (belonging to a particular date) that registered a negative sentiment")
                st.markdown(
                    "%_of_mixed_sentiments: Percentage of tweets (belonging to a particular date) that registered a mixed sentiment")
                st.markdown(
                    "%_of_neutral_sentiments: Percentage of tweets (belonging to a particular date) that registered a neutral sentiment")

            correlation_values = st.expander('Correlation values')
            with correlation_values:

                st.info("🛈 Note that if any of correlation values are displayed as <NA>, then most probably, it is because one of the variables was constant during the selected time period!")

                data1 = {'Correlation values': [
                    corr_df1.iat[0, 8], corr_df2.iat[0, 8], corr_df3.iat[0, 8]]}
                data2 = {'Correlation values': [
                    corr_df1.iat[0, 6], corr_df2.iat[0, 6], corr_df3.iat[0, 6]]}
                data3 = {'Correlation values': [
                    corr_df1.iat[0, 7], corr_df2.iat[0, 7], corr_df3.iat[0, 7]]}
                data4 = {'Correlation values': [
                    corr_df1.iat[0, 5], corr_df2.iat[0, 5], corr_df3.iat[0, 5]]}

                cus_df1 = pd.DataFrame.from_dict(data1, orient='index',
                                                 columns=['Pearson', 'Kendall', 'Spearman'])
                cus_df2 = pd.DataFrame.from_dict(data2, orient='index',
                                                 columns=['Pearson', 'Kendall', 'Spearman'])
                cus_df3 = pd.DataFrame.from_dict(data3, orient='index',
                                                 columns=['Pearson', 'Kendall', 'Spearman'])
                cus_df4 = pd.DataFrame.from_dict(data4, orient='index',
                                                 columns=['Pearson', 'Kendall', 'Spearman'])

                st.markdown(
                    "Correlation between Positive Sentiments and Stringency Index")
                st.table(cus_df1)
                st.markdown(
                    "Correlation between Negative Sentiments and Stringency Index")
                st.table(cus_df2)
                st.markdown(
                    "Correlation between Neutral Sentiments and Stringency Index")
                st.table(cus_df3)
                st.markdown(
                    "Correlation between Mixed Sentiments and Stringency Index")
                st.table(cus_df4)

            if (country_selected == "Hong Kong"):
                st.markdown(
                    "Sorry, we were unable to find covid reproduction rate in Hong Kong!")
            else:
                fig2 = make_subplots(rows=1, cols=1)
                fig2.add_scatter(
                    x=df_filtered["date"], y=df_filtered["reproduction_rate"], mode='lines', name="reproduction_rate")
                fig2.add_scatter(x=df_filtered["date"], y=[
                    val / 100 for val in df_filtered["%_of_positive_sentiments"]],
                    mode='lines', name="proportion_of_positive_sentiments")
                fig2.add_scatter(x=df_filtered["date"], y=[
                    val / 100 for val in df_filtered["%_of_negative_sentiments"]],
                    mode='lines', name="proportion_of_negative_sentiments")
                fig2.add_scatter(x=df_filtered["date"], y=[
                    val / 100 for val in df_filtered["%_of_mixed_sentiments"]],
                    mode='lines', name="proportion_of_mixed_sentiments")
                fig2.add_scatter(x=df_filtered["date"], y=[
                    val / 100 for val in df_filtered["%_of_neutral_sentiments"]],
                    mode='lines', name="proportion_of_neutral_sentiments")

                fig2.update_layout(
                    title="<b>Chart 2: Reproduction Rate vs Time & Proportions of Sentiments vs Time</b>",
                    xaxis_title="Date",
                    yaxis_title="Values",
                    font=dict(
                        size=13
                    )
                )

                st.plotly_chart(fig2)

                st.info(
                    "🛈 Click on the respective variable legend (on the right of the chart) to select or deselect it.")

                metrics_meaning = st.expander('What does the metrics mean?')
                with metrics_meaning:
                    st.markdown(
                        "reproduction_rate: Real-time estimate of the effective reproduction rate (R) of COVID-19")
                    st.markdown(
                        "%_of_positive_sentiments: Percentage of tweets (belonging to a particular date) that registered a positive sentiment")
                    st.markdown(
                        "%_of_negative_sentiments: Percentage of tweets (belonging to a particular date) that registered a negative sentiment")
                    st.markdown(
                        "%_of_mixed_sentiments: Percentage of tweets (belonging to a particular date) that registered a mixed sentiment")
                    st.markdown(
                        "%_of_neutral_sentiments: Percentage of tweets (belonging to a particular date) that registered a neutral sentiment")

                correlation_values = st.expander('Correlation values')
                with correlation_values:
                    # st.table(corr_df1)
                    # st.table(corr_df2)
                    # st.table(corr_df3)

                    st.info("🛈 Note that if any of correlation values are displayed as <NA>, then most probably, it is because one of the variables was constant during the selected time period!")

                    data1 = {'Correlation values': [
                        corr_df1.iat[1, 8], corr_df2.iat[1, 8], corr_df3.iat[1, 8]]}
                    data2 = {'Correlation values': [
                        corr_df1.iat[1, 6], corr_df2.iat[1, 6], corr_df3.iat[1, 6]]}
                    data3 = {'Correlation values': [
                        corr_df1.iat[1, 7], corr_df2.iat[1, 7], corr_df3.iat[1, 7]]}
                    data4 = {'Correlation values': [
                        corr_df1.iat[1, 5], corr_df2.iat[1, 5], corr_df3.iat[1, 5]]}

                    cus_df1 = pd.DataFrame.from_dict(data1, orient='index',
                                                     columns=['Pearson', 'Kendall', 'Spearman'])
                    cus_df2 = pd.DataFrame.from_dict(data2, orient='index',
                                                     columns=['Pearson', 'Kendall', 'Spearman'])
                    cus_df3 = pd.DataFrame.from_dict(data3, orient='index',
                                                     columns=['Pearson', 'Kendall', 'Spearman'])
                    cus_df4 = pd.DataFrame.from_dict(data4, orient='index',
                                                     columns=['Pearson', 'Kendall', 'Spearman'])

                    st.markdown(
                        "Correlation between Positive Sentiments and Reproduction Rate")
                    st.table(cus_df1)
                    st.markdown(
                        "Correlation between Negative Sentiments and Reproduction Rate")
                    st.table(cus_df2)
                    st.markdown(
                        "Correlation between Neutral Sentiments and Reproduction Rate")
                    st.table(cus_df3)
                    st.markdown(
                        "Correlation between Mixed Sentiments and Reproduction Rate")
                    st.table(cus_df4)

        else:
            st.markdown(
                "Please expand the tabs below to find out more about the project!")

            see_intro = st.expander('Project Details')
            with see_intro:
                st.markdown("Over the recent years, the COVID-19 pandemic has caused serious disruptions all over the world. The pandemic has caused a shear negative impact on the economy and peoples’ livelihoods. Given the increasing vaccine rollup and take-up rate along with the growing evidence about the virus slowly becoming weaker, many governments have become eager to ease the curbs that they once implemented. The motivation is to boost their economies, return to the pre-pandemic normal and as a result improve the lives of their people. However, lifting the restrictions usually causes the cases to rise and more to get severely ill or worse, die. As a result, there are mixed feelings among people whether to ease the restrictions. In the case of Singapore, when the government announced that it will be postponing the streamlined measures in view of the rising cases and the resulting pressure on the healthcare workers, there were again mixed reactions with some applauding the government and the others being disappointed.")
                st.markdown("Therefore, we aim to investigate peoples’ general sentiment, especially towards the easing of COVID-19 restrictions brought by their governments. To achieve such a purpose, we would like to make well use of COVID-19 related tweets since Twitter is undoubtedly one of the best platforms to obtain data regarding people’s general feelings given that many use it to regularly post their opinions and let their voices be heard. From the start of 2022 onwards, Twitter has 217 million monetizable daily active users and the number of tweets sent per day reaches approximately 500 million. The gigantic number of tweets that are made publicly available every day enables better predictions and more accurate interpretation of the public’s sentiments around the globe. Moreover, there are plentiful well-established APIs that make it practical to conduct sentiment analysis on these tweets.")

            see_dataset_used = st.expander('Dataset Used')
            with see_dataset_used:
                st.markdown("The primary dataset type that we will be collecting, is recent twitter tweets from January 1st 2022 to Feb 28th 2022. For this, we will be extracting tweets using a hydrator tool from a GitHub repository (https://github.com/echen102/COVID-19-TweetIDs) that contains an archive of tweet IDs that are associated with COVID-19, filtered by specific periods. The dataset size after hydration can range from 20-50 GB for each-day data (about 2TB data across a two-month period), indicating the need for large storage space and efficient processing power. We also had a decision to make whether to include only English tweets, or to include tweets from other languages. We weighed the pros of both options, where with English tweets we would be able to account for accuracy in tone, while having multiple languages provides more representation of sentiments. In the end, we decided to only include English tweets as we valued the accuracy in our sentiment analysis greater than the representativeness of the sentiments.")
                st.markdown("Next up, we are also interested to find how the restrictions in these countries changed over the period to understand the change in sentiments. For that, we will be using the COVID-19 Stringency Index, a composite measure based on nine response indicators that measures the strictness of the restrictions in a country. We will download the data from the index’s website. From this website, we will also be extracting other covid-related statistics of a country such as the country's covid reproduction rate.")

            see_raw_data = st.expander(
                'Raw Data (that was ingested, processed and aggregated by us)')
            with see_raw_data:
                st.dataframe(data=df.reset_index(drop=True))

    else:
        st.sidebar.subheader(
            "Now, select the countries that you want to compare...")
        selected_countries = st.sidebar.multiselect(
            "Select and deselect the countries that you want to compare. You can clear the current selection by clicking the corresponding x-button on the right", countries, default=countries)

        st.sidebar.subheader("Now, select a time frame...")
        start_date = st.sidebar.date_input(
            "Select a start date:",
            min_value=dt.date(2022, 1, 1),
            value=dt.date(2022, 1, 1),
            max_value=dt.date(2022, 2, 28))
        end_date = st.sidebar.date_input(
            "Select a end date:",
            min_value=dt.date(2022, 1, 1),
            value=dt.date(2022, 1, 1),
            max_value=dt.date(2022, 2, 28))

        if start_date <= end_date:
            pass
        else:
            st.error('Error: End date must fall after start date.')

        if st.sidebar.button('Create Visualization'):
            #df_filtered = df.loc[df['location'] == selected_countries[0]]
            #df_filtered = df_filtered.loc[(df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)]
            #fig = px.line(df_filtered, x="date", y=df.columns[2:8])

            fig = make_subplots(rows=1, cols=1)

            for i in range(0, len(selected_countries)):
                country = selected_countries[i]
                df_filtered_i = df.loc[df['location'] == country]
                df_filtered_i = df_filtered_i.loc[(df_filtered_i['date'] >= start_date) & (
                    df_filtered_i['date'] <= end_date)]
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["stringency_index"],
                                mode='lines', name="{cname}-stringency_index".format(cname=country))
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["reproduction_rate"],
                                mode='lines', name="{cname}-reproduction_rate".format(cname=country))
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["new_deaths_per_million"],
                                mode='lines', name="{cname}-new_deaths_per_million".format(cname=country))
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["%_of_positive_sentiments"],
                                mode='lines', name="{cname}-%_of_positive_sentiments".format(cname=country))
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["%_of_negative_sentiments"],
                                mode='lines', name="{cname}-%_of_negative_sentiments".format(cname=country))
                fig.add_scatter(x=df_filtered_i["date"], y=df_filtered_i["%_of_mixed_sentiments"],
                                mode='lines', name="{cname}-%_of_mixed_sentiments".format(cname=country))

            st.plotly_chart(fig)

            st.markdown(
                "Click on the respective variable legend (on the right of the chart) to select and deselect the variable")
            st.markdown(
                "Expand the tab below to better understand what each metrics/variables mean")

            metrics_meaning = st.expander('What does the metrics mean?')
            with metrics_meaning:
                st.markdown("stringency_index: Composite measure based on nine response indicators including school closures, workplace closures, and travel bans, rescaled to a value from 0 to 100 (100 = strictest)")
                st.markdown(
                    "reproduction_rate: Real-time estimate of the effective reproduction rate (R) of COVID-19")
                st.markdown(
                    "new_deaths_per_million: New deaths attributed to COVID-19 per 1,000,000 people. Counts can include probable deaths, where reported")

        else:
            see_intro = st.expander('Project Details')
            with see_intro:
                st.markdown("Over the recent years, the COVID-19 pandemic has caused serious disruptions all over the world. The pandemic has caused a shear negative impact on the economy and peoples’ livelihoods. Given the increasing vaccine rollup and take-up rate along with the growing evidence about the virus slowly becoming weaker, many governments have become eager to ease the curbs that they once implemented. The motivation is to boost their economies, return to the pre-pandemic normal and as a result improve the lives of their people. However, lifting the restrictions usually causes the cases to rise and more to get severely ill or worse, die. As a result, there are mixed feelings among people whether to ease the restrictions. In the case of Singapore, when the government announced that it will be postponing the streamlined measures in view of the rising cases and the resulting pressure on the healthcare workers, there were again mixed reactions with some applauding the government and the others being disappointed.")
                st.markdown("Therefore, we aim to investigate peoples’ general sentiment, especially towards the easing of COVID-19 restrictions brought by their governments. To achieve such a purpose, we would like to make well use of COVID-19 related tweets since Twitter is undoubtedly one of the best platforms to obtain data regarding people’s general feelings given that many use it to regularly post their opinions and let their voices be heard. From the start of 2022 onwards, Twitter has 217 million monetizable daily active users and the number of tweets sent per day reaches approximately 500 million. The gigantic number of tweets that are made publicly available every day enables better predictions and more accurate interpretation of the public’s sentiments around the globe. Moreover, there are plentiful well-established APIs that make it practical to conduct sentiment analysis on these tweets.")

            see_raw_data = st.expander('Raw Data')
            with see_raw_data:
                st.dataframe(data=df.reset_index(drop=True))


if __name__ == '__main__':
    run()
