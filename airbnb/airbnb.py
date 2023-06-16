import pandas as pd
import numpy as np
from sklearn import model_selection as ms
from sklearn import preprocessing as pp

pd.set_option('mode.chained_assignment', None)

class Airbnb:

    def load_data(self):
        df_sessions = pd.read_parquet('./data/sessions.parquet')
        return df_sessions


    def transform_data(self, df, df_sessions):
        #==================Training================
        #age
        age_mean = df['age'].mean()
        df['age'] = df['age'].fillna(age_mean)

        # first_affiliate_tracked
        df['first_affiliate_tracked'].dropna(inplace=True)

        #==================Sessions==============
        df_sessions.dropna(inplace = True)

        #date_account_created
        df['date_account_created'] = pd.to_datetime(df['date_account_created'])

        #timestamp_first_active
        df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S')
        # date_first_booking - not available
        df.drop('date_first_booking', axis = 1, inplace = True)

        #age
        df['age'] = df['age'].astype('int64')

        #Filter
        df = df[(df['age']>15) & (df['age']<100)]

        return df, df_sessions

    def feature_engineering(self, df, df_sessions):

        df['first_active'] = pd.to_datetime(df['timestamp_first_active'].dt.strftime('%Y-%m-%d'))

        #time between account created and first active
        df['days_from_active_to_account_created'] = (df['date_account_created'] - df['first_active']).dt.days

        #year  of first active
        df['year_first_active'] = df['first_active'].dt.year

        #month of first active
        df['month_first_active'] = df['first_active'].dt.month

        #day of first active
        df['day_first_active'] = df['first_active'].dt.day

        #day of week of first active
        df['day_of_week_first_active'] = df['first_active'].dt.dayofweek

        #week of year of first active
        df['week_of_year_first_active'] = df['first_active'].dt.isocalendar().week

        #year  of account created
        df['year_account_created'] = df['date_account_created'].dt.year

        #month of account created
        df['month_account_created'] = df['date_account_created'].dt.month

        #day of account created
        df['day_account_created'] = df['date_account_created'].dt.day

        #day of week of account created
        df['day_of_week_account_created'] = df['date_account_created'].dt.dayofweek

        #week of year of account created
        df['week_of_year_account_created'] = df['date_account_created'].dt.isocalendar().week

        # n_clicks
        n_clicks = df_sessions[df_sessions['action_type']=='click'].groupby('user_id').agg(n_clicks = ('user_id', 'count')).reset_index()
        df = pd.merge(df, n_clicks.rename(columns = {'user_id' : 'id'}), on ='id', how='left')
        df['n_clicks'].fillna(0, inplace=True)

        n_reviews = df_sessions[df_sessions['action']=='reviews'].groupby('user_id').agg(n_reviews = ('user_id', 'count')).reset_index()
        df = pd.merge(df, n_reviews.rename(columns = {'user_id' : 'id'}), on ='id', how='left')
        df['n_reviews'].fillna(0, inplace=True)

        return df

    def data_preprocessing(self, df):
        
        # language to binary, either is english or not
        df['language_en'] = np.where(df['language']=='en', 1, 0)

        # signup to binary, either is web or not
        df['signup_on_web'] = np.where(df['signup_app']== 'Web', 1, 0)

        # first_affiliate_tracked to binary, either is tracked or not
        df['tracked'] = np.where(df['first_affiliate_tracked']=='untracked', 0, 1)

        #binary features from first_device_type
        df['first_device_apple'] = np.where(df['first_device_type'].isin(['Mac Desktop', 'iPhone', 'iPad']), 1 ,0)
        df['first_device_desktop'] = np.where(df['first_device_type'].isin(['Mac Desktop', 'Desktop', 'Windows Desktop']), 1, 0)

        # frequency encoding
        affiliate_channel_frequency_encoding = df['affiliate_channel'].value_counts(normalize=True)
        df['affiliate_channel'] = df['affiliate_channel'].map(affiliate_channel_frequency_encoding)

        affiliate_provider_frequency_encoding = df['affiliate_provider'].value_counts(normalize=True)
        df['affiliate_provider'] = df['affiliate_provider'].map(affiliate_provider_frequency_encoding)

        first_browser_frequency_encoding = df['first_browser'].value_counts(normalize=True)
        df['first_browser'] = df['first_browser'].map(first_browser_frequency_encoding)

        # Rescaling
        columns_to_rescale = [
        "age",
        "signup_flow",
        "n_reviews",
        "n_clicks"
        ]

        scaler = pp.MinMaxScaler()
        df[columns_to_rescale] = scaler.fit_transform(df[columns_to_rescale])

         # month_account_created
        df['month_account_created_sin'] = df['month_account_created'].apply( lambda x: np.sin( x * (2*np.pi/12 ) ) )
        df['month_account_created_cos'] = df['month_account_created'].apply( lambda x: np.cos( x * (2*np.pi/12 ) ) )

        # week_account_created
        df['week_account_created_sin'] = df['week_of_year_account_created'].apply( lambda x: np.sin( x * (2*np.pi/52 ) ) )
        df['week_account_created_cos'] = df['week_of_year_account_created'].apply( lambda x: np.cos( x * (2*np.pi/52 ) ) )

        # day_account_created
        df['day_account_created_sin'] = df['day_account_created'].apply( lambda x: np.sin( x * (2*np.pi/30 ) ) )
        df['day_account_created_cos'] = df['day_account_created'].apply( lambda x: np.cos( x * (2*np.pi/30 ) ) )

        # day_of_week_account_created
        df['day_of_week_account_created_sin'] = df['day_of_week_account_created'].apply( lambda x: np.sin( x * (2*np.pi/7 ) ) )
        df['day_of_week_account_created_cos'] = df['day_of_week_account_created'].apply( lambda x: np.cos( x * (2*np.pi/7 ) ) )

        X = df[['age', 'signup_flow', 'affiliate_channel', 'affiliate_provider',
                 'first_browser', 'n_clicks', 'n_reviews',
                 'language_en', 'signup_on_web', 'tracked', 'first_device_apple',
                 'first_device_desktop', 'month_account_created_sin',
                 'month_account_created_cos', 'week_account_created_sin',
                 'week_account_created_cos', 'day_account_created_sin',
                 'day_account_created_cos', 'day_of_week_account_created_sin',
                 'day_of_week_account_created_cos']]

        return X

    def predict(self, model, X_test):  
        return model.predict_proba(X_test)