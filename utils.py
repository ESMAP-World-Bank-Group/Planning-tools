# This script is used to fetch solar or wind power data for multiple locations using the Renewables Ninja API.
# The API requires an API token, which you can get by signing up at https://www.renewables.ninja/.
# The API allows you to fetch hourly solar or wind power data for a given location, time period, and power system configuration.
# The data is returned in JSON format, which can be easily converted to a Pandas DataFrame.
# The function get_renewable_data() fetches the data for multiple locations and returns a dictionary of results for each location.
# The function also handles rate limiting by waiting for a minute if the rate limit is hit.
# The data is then saved to a CSV file for further analysis.
# Author: Lucas Vivier: lvivier@worldbank.org

import requests
import pandas as pd
import time
import os
import numpy as np
import gams.transfer as gt
import subprocess
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Tuple, Union
import matplotlib.pyplot as plt
import calendar
import warnings
import logging
import sys
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


logging.basicConfig(level=logging.WARNING)  # Configure logging level
logger = logging.getLogger(__name__)

API_TOKEN = '2deb093cdece49f3b316c98687150421ee425566'
nb_days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], index=range(1, 13))


def get_renewable_data(power_type, locations, start_date, end_date, api_token=API_TOKEN, dataset="merra2", capacity=1,
                       system_loss=0.1, height=100, tracking=0, tilt=35, azim=180,
                       turbine='Gamesa+G114+2000', local_time='true'):
    """Fetch solar or wind power data for multiple locations using the Renewables Ninja API.

    Args:
    - api_token (str): Your Renewables Ninja API token.
    - power_type (str): 'solar' or 'wind'.
    - locations (list of tuples): List of locations as (latitude, longitude).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - dataset (str): Dataset to use ('merra2' or 'sarah' for solar). Defaults to 'merra2'.
    - capacity (float): Capacity of the power system. Defaults to 1.
    - system_loss (float): System losses (for solar). Defaults to 0.1.
    - height (float): Turbine height (for wind). Defaults to 100.
    - tracking (int): Tracking type for solar (0 for fixed, 1 for single-axis). Defaults to 0.
    - tilt (float): Tilt angle for solar panels. Defaults to 35.
    - azim (float): Azimuth angle for solar panels. Defaults to 180.
    - turbine (str): Turbine type for wind. Defaults to 'Vestas+V80+2000'.
    - local_time (bool): Whether to return data in local time. Defaults to True.

    Returns:
    - dict: A dictionary of results for each location.
    """
    base_url = 'https://www.renewables.ninja/api/data/'

    # Set headers for the API request
    headers = {
        'Authorization': f'Token {api_token}'
    }

    # Track requests per minute
    requests_made_minute = 0
    requests_tot = 0
    minute_start_time = time.time()

    all_data = []

    # Iterate over locations
    for country, (lat, lon) in locations.items():
    # Build the request URL based on the power type (solar or wind)
        if power_type == 'solar':
            url = f"{base_url}pv?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&system_loss={system_loss}&tracking={tracking}&tilt={tilt}&azim={azim}&local_time={local_time}&format=json"
        elif power_type == 'wind':
            url = f"{base_url}wind?lat={lat}&lon={lon}&date_from={start_date}&date_to={end_date}&dataset={dataset}&capacity={capacity}&height={height}&turbine={turbine}&local_time={local_time}&format=json"
        else:
            raise ValueError("Invalid power_type. Choose either 'solar' or 'wind'.")

        # Send the request
        response = requests.get(url, headers=headers, verify=False)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            # Append the data along with location details
            for timestamp, values in data['data'].items():
                row = {
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': pd.to_datetime(int(timestamp) / 1000, unit='s'),
                    # 'local_time': pd.to_datetime(int(local_time) / 1000, unit='s'),
                    **values  # Unpack all the power generation data at that timestamp
                }
                all_data.append(row)
        else:
            print(f"Error fetching data for location ({lat}, {lon}): {response.status_code}")

        # Track the request
        requests_made_minute += 1
        requests_tot += 1

        # If we hit 6 requests within a minute, we wait for the remaining time of that minute
        if requests_made_minute >= 6:
            print('Waiting for a minute to not hit the API rate limit...')
            elapsed_time = time.time() - minute_start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the counter and timer
            requests_made_minute = 0
            minute_start_time = time.time()

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(all_data)
    df.rename(columns={'electricity': power_type}, inplace=True)
    return df, requests_tot


def get_years_renewables(locations, power_type, start_year, end_year, start_day='01-01', end_day='12-31',
                         turbine='Gamesa+G114+2000', name_data='data', output='data'):
    """Get the renewable data for multiple years.

    Parameters
    ----------
    locations: list
        List of locations as (latitude, longitude).
    power_type: str
        'solar' or 'wind'.
    start_year: int
        Start year.
    end_year: int
        End year.
    start_day: str, optional, default '01-01'
        Start day.
    end_day: str, optional, default '12-31'
        End day.
    turbine: str, optional, default 'Gamesa+G114+2000'
        Turbine type for wind.
    name_data: str, optional, default 'data'
        Name of the data.

    Returns
    -------
    results_concat: pd.DataFrame
        DataFrame with the energy data.
    """

    results = {}
    hour_start_time = time.time()
    requests_tot = 0
    for year in range(start_year, end_year):

        start_date = '{}-{}'.format(year, start_day)
        end_date = '{}-{}'.format(year, end_day)

        # Call the function to get data
        data, requests_made = get_renewable_data(power_type, locations, start_date, end_date, turbine=turbine)
        requests_tot += requests_made

        if data.empty:
            print("No data for year {}".format(year))
            continue
        else:
            print("Getting data {} for year {}".format(power_type, year))
            data.set_index(['latitude', 'longitude'], inplace=True)
            data['local_time'] = pd.to_datetime(data['local_time'], utc=True)
            data['season'] = data['local_time'].dt.month
            data['day'] = data['local_time'].dt.day
            data['hour'] = data['local_time'].dt.hour
            data['time_only'] = data['local_time'].dt.strftime('%m-%d %H:%M:%S')
            data.set_index(['time_only', 'season', 'day', 'hour'], inplace=True, append=True)
            data = data.loc[:, power_type]

            results.update({year: data})

        # If we hit 50 requests within an hour, we wait for the remaining time of that hour before continuing the requests
        if requests_tot >= 36:
            print('Waiting for an hour to not hit the API rate limit...')
            elapsed_time = time.time() - hour_start_time
            if elapsed_time < 3600:
                sleep_time = 3600 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the counter and timer
            requests_tot = 0
            hour_start_time = time.time()

    results_concat = pd.concat(results, axis=1)
    results_concat.to_csv(os.path.join(output, 'data_{}_{}.csv'.format(name_data, power_type)))
    return results_concat


def find_representative_year(df, method='average_profile'):
    """Find the representative year.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the energy data.
    method: str, optional, default 'average_profile'
        Method to find the representative year.

    Returns
    -------
    repr_year: int
        Representative year.
    """
    dict_info, repr_year = {}, None
    if method == 'average_cf':
        # Find representative year
        temp = df.sum().round(3)
        # get index closest (not equal) to the median in the temp
        repr_year = (np.abs(temp - temp.median())).idxmin()
        print('Representative year {}'.format(repr_year), 'with production of {:.0%}'.format(temp[repr_year] / 8760))
        print('Standard deviation of {:.3%}'.format(temp.std() / 8760))
        dict_info.update({'year_repr': repr_year, 'std': temp.std(), 'median': temp.median()})

    elif method == 'average_profile':
        average_profile = df.mean(axis=1)
        deviations = {}
        for year in df.columns:
            deviations[year] = np.sum(np.abs(df[year] - average_profile))
        repr_year = min(deviations, key=deviations.get)
    else:
        raise ValueError("Invalid method. Choose either 'representative_year' or 'average_profile'.")

    return repr_year


def format_data_energy(filenames, locations):
    """Format the data for the energy from .csv files.

    Parameters
    ----------
    filenames: dict
        Dictionary with the filenames.

    Returns
    -------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    """
    # Find the representative year

    df_energy = {}
    for key, item in filenames.items():
        filename, reading = item[0], item[1]

        # Extract from the data results
        if reading == 'renewable_ninja':
            df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3, 4, 5])
            df = df.reset_index()
            df['tech'] = key
            df['zone'] = list(zip(df['latitude'], df['longitude']))  # Convert lat/lon to tuples
            df['zone'] = df['zone'].map(locations)
            df = df.drop(columns=['latitude', 'longitude', 'time_only']).set_index(['zone', 'season', 'day', 'hour', 'tech'])

        elif reading == 'standard':
            df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3])
            df = df.reset_index()
            df['tech'] = key
            df = df.set_index(['zone', 'season', 'day', 'hour', 'tech'])

        else:
            raise ValueError('Unknown reading. Only implemented for: renewable_ninja, standard.')

        df_energy.update({key: df})

    df_energy = pd.concat(df_energy.values(), ignore_index=False)

    repr_year = find_representative_year(df_energy)
    print('Representative year {}'.format(repr_year))
    df_energy = df_energy.loc[:, repr_year].unstack('tech').reset_index()
    df_energy = df_energy.sort_values(by=['zone', 'season', 'day', 'hour'], ascending=True)

    # If 2/29, remove it
    if len(df_energy.season.unique()) == 12:  # season expressed as months
        df_energy = df_energy[~((df_energy['season'] == 2) & (df_energy['day'] == 29))]

    if df_energy.isna().any().any():
        print('Warning: NaN values in the DataFrame')

    keys_to_merge = ['PV', 'Wind', 'Load', 'ROR']
    keys_to_merge = [i for i in keys_to_merge if i in df_energy.keys()]

    print('Annual capacity factor (%):', df_energy.groupby('zone')[keys_to_merge].mean().reset_index())
    if len(df_energy.zone.unique()) > 1:  # handling the case with multiple zones to rename columns
        df_energy = df_energy.set_index(['zone', 'season', 'day', 'hour']).unstack('zone')
        df_energy.columns = ['_'.join([idx0, idx1]) for idx0, idx1 in df_energy.columns]
        df_energy = df_energy.reset_index()
    else:
        df_energy = df_energy.drop('zone', axis=1)
    return df_energy


def cluster_data_new(df_energy, n_clusters=10, columns=None):
    """
    Perform KMeans clustering on daily energy data to identify representative clusters by season.

    This function applies KMeans clustering to energy data grouped by season, where each data point
    represents the sum of values for a single day. It enables identification of clustered daily profiles
    (e.g., for PV, Wind, Load) separately for each season.

    Parameters:
        df_energy (pd.DataFrame):
            DataFrame containing energy-related features with columns ['season', 'day', 'hour', 'PV', 'Wind', 'Load'].
        n_clusters (int, optional):
            Number of clusters to create. Defaults to 10.
        columns (list, optional):
            List of feature columns used for clustering. If None, all columns except ['season', 'day', 'hour'] are used.

    Returns:
        tuple:
            - pd.DataFrame: Original DataFrame with assigned cluster labels.
            - pd.DataFrame: Representative days closest to cluster centroids.
            - pd.DataFrame: Cluster centroids with associated probabilities.
    """

    # Select relevant columns for clustering
    if columns is None:
        columns = [i for i in df_energy.columns if i not in ['season', 'day', 'hour']]

    # Find closest days to centroids
    def find_closest_days(df, centroids, features):
        closest_days = []
        for i, centroid in enumerate(centroids):
            # Compute Euclidean distance between each row and the centroid
            distances = np.linalg.norm(df[features] - centroid, axis=1)
            closest_idx = distances.argmin()  # Find index of closest day
            closest_days.append(df.iloc[closest_idx])  # Store closest day
        closest_days = pd.DataFrame(closest_days)
        closest_days.index = range(len(centroids))
        return closest_days

    # Compute cluster probabilities
    def assign_probabilities(labels: np.ndarray, n_clusters: int) -> pd.Series:
        """
        Calculate probabilities for each cluster based on frequency of occurrence.
        """
        unique, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return pd.Series(probabilities, index=range(n_clusters))

    df_closest_days = []
    centroids_df = []
    df_tot = df_energy.copy()
    for season, df_season in df_tot.groupby('season'):
        df_season_cluster = df_season.copy()
        df_season_cluster = df_season_cluster.groupby(['season', 'day'])[columns].sum().reset_index()

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_season_cluster['Cluster'] = kmeans.fit_predict(df_season_cluster[columns])
        df_tot = df_tot.merge(df_season_cluster[['season', 'day', 'Cluster']], on=['season', 'day'], how='left',
                              suffixes=('', '_temp'))

        # Fill only missing values in df_tot['Cluster']
        if 'Cluster_temp' in df_tot.columns:
            df_tot['Cluster'] = df_tot['Cluster'].combine_first(df_tot['Cluster_temp'])
            df_tot = df_tot.drop(['Cluster_temp'], axis=1)

        # Extract cluster centers
        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns, index=range(n_clusters))
        cluster_probabilities = assign_probabilities(df_season_cluster['Cluster'].values, n_clusters)
        centroids = pd.concat([cluster_probabilities.to_frame().rename(columns={0: 'probability'}), cluster_centers],
                              axis=1)
        centroids['season'] = season
        centroids = centroids.reset_index().rename(columns={'index': 'Cluster'})
        centroids_df.append(centroids)

        df_closest_days.append(find_closest_days(df_season_cluster, cluster_centers.values, columns))

    df_closest_days, centroids_df = pd.concat(df_closest_days, axis=0, ignore_index=True), pd.concat(centroids_df,
                                                                                                     axis=0,
                                                                                                     ignore_index=True)
    df_closest_days = df_closest_days.merge(centroids_df[['Cluster', 'probability', 'season']],
                                            on=['Cluster', 'season'], how='left')
    return df_tot, df_closest_days, centroids_df


def select_representative_series_hierarchical(path_data_file: str, n: int, method: str = 'ward',
                                              metric: str = 'euclidean', scale: bool = True,
                                              scale_method: str = 'standard'):
    """
    Reduce dimensionality of time series features using hierarchical clustering, keeping only real series.

    Parameters:
        path_data_file (str):
            Path to the dataframe with columns as different time series (e.g., 'Wind_ZM', 'PV_BW', 'Corr_Load_TZ__PV_MZ', etc.)
            and rows as time steps (e.g., hours in the season).
        n (int):
            Number of representative time series to keep.
        method (str):
            Linkage method for hierarchical clustering (e.g., 'ward', 'average', 'complete').
        metric (str):
            Distance metric (e.g., 'euclidean', 'cosine', 'correlation').

    Returns:
        list:
            Names of selected time series (columns) to retain.
        pd.DataFrame:
            Reduced DataFrame with only the selected columns.
    """
    path_data_file = os.path.join(os.getcwd(), path_data_file)
    df = pd.read_csv(path_data_file, index_col=[0,1,2])

    # Transpose to get features as rows
    series_matrix = df.T.values  # shape: (n_series, n_timesteps)
    series_names = df.columns.tolist()

    # Optional scaling
    if scale:
        if scale_method == 'standard':
            scaler = StandardScaler()
        elif scale_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("scale_method must be 'standard' or 'minmax'")
        series_matrix = scaler.fit_transform(series_matrix)

    # Compute pairwise distance
    distance_matrix = pdist(series_matrix, metric=metric)

    # Hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)

    # Cut the dendrogram into n clusters
    cluster_labels = fcluster(linkage_matrix, t=n, criterion='maxclust')

    # For each cluster, find the series closest to the centroid
    selected_series = []
    for cluster_id in range(1, n + 1):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_vectors = series_matrix[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)

        # Compute distance to centroid
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        selected_series.append(series_names[closest_idx])

    path_data_file_selection = path_data_file.split('.csv')[0]
    path_data_file_selection = f'{path_data_file_selection}_selection.csv'

    df[selected_series].to_csv(path_data_file_selection, index=True)
    print('File saved at:', path_data_file_selection)

    return selected_series, df[selected_series], path_data_file_selection


def get_special_days_clustering(df_closest_days, df_tot, threshold=0.07):
    """
    Identify special days based on clustering results.

    This function selects extreme days (e.g., lowest PV, lowest Wind, highest Load) as centroids of clusters,
    while ensuring that clusters representing a large share of the data are excluded.

    Parameters:
        df_closest_days (pd.DataFrame):
            DataFrame of closest representative days to cluster centroids.
        df_tot (pd.DataFrame):
            Original dataset with all time series data.
        threshold (float, optional):
            Probability threshold for excluding large clusters. Defaults to 0.07.

    Returns:
        tuple:
            - pd.DataFrame: Special days with associated weights.
            - pd.DataFrame: Updated df_tot with special days removed.
    """
    df_tot = df_tot.copy()
    special_days = []
    indices_to_remove = set()
    for season, df_season in df_closest_days.groupby('season'):
        # first special day is based on minimum PV production across all zones
        add_special_days(feature='PV', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='min', threshold=0.07)

        # second special day is based on minimum Wind production across all zones
        add_special_days(feature='Wind', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='min', threshold=0.07)

        # third special day is based on maximum peak demand across all zones
        add_special_days(feature='Load', df=df_season, df_days=df_tot, season=season, special_days=special_days,
                         indices_to_remove=indices_to_remove, rule='max', threshold=0.07)


    # Convert special days to DataFrame
    df_special_days = pd.DataFrame(special_days)
    df_special_days = df_special_days.drop_duplicates()
    df_tot = df_tot.drop(index=indices_to_remove)  # we remove the days corresponding to clusters which have been included as special days
    df_tot = df_tot.drop(columns=['Cluster'])  # no longer needed

    return df_special_days, df_tot


def add_special_days(feature, df, df_days, season, special_days, indices_to_remove, rule='min', threshold=0.07):
    """
    Identifies and adds a representative 'special day' from a given season based on extreme values
    of a selected feature (e.g., PV, Load, Wind) across multiple zones.

    The function looks for the day within a season that corresponds to the most extreme value
    (minimum or maximum depending on the `rule`) of the summed feature across all zones, excluding
    high-probability clusters (based on the `threshold`). If the most extreme day is already present
    in `special_days`, the function iteratively selects the next most extreme day that hasn’t been used.

    Parameters:
        feature (str):
            The feature to analyze (e.g., 'PV', 'Wind', 'Load').
        df (pd.DataFrame):
            DataFrame containing cluster probabilities and feature values.
        df_days (pd.DataFrame):
            Complete dataset with all time series data.
        season (str):
            Current season being analyzed.
        special_days (list):
            List of previously identified special days.
        indices_to_remove (set):
            Set to track days to be removed from df_days.
        rule (str, optional):
            Whether to select the 'min' or 'max' extreme value. Defaults to 'min'.
        threshold (float, optional):
            Probability threshold for excluding large clusters. Defaults to 0.07.

    Returns:
        list: Updated list of special days.
    """
    assert rule in ['min', 'max'], "Rule for selecting cluster should be either 'min' or 'max'."
    columns = [col for col in df.columns if feature in col]
    df = df.copy()

    if len(columns) > 0:
        df = df[df['probability'] < threshold]
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)

        def find_next_special_day(df, special_days, rule):
            for i in range(len(df)):  # Iterate over all ranked rows
                cluster = df.nsmallest(i + 1, feature).iloc[-1:] if rule == 'min' else df.nlargest(i + 1, feature).iloc[-1:]
                special_day = tuple(cluster[['season', 'day']].values[0])
                if special_day not in {entry['days'] for entry in special_days}:
                    return cluster
            return None  # This should never be reached unless all days are special (unlikely)

        # Find the most extreme day based on rule
        cluster = df.nsmallest(1, feature) if rule == 'min' else df.nlargest(1, feature)
        special_day = tuple(cluster[['season', 'day']].values[0])

        if special_day in {entry['days'] for entry in special_days}:
            # If already included, find the next available extreme day
            cluster = find_next_special_day(df, special_days, rule)

        if cluster is not None:
            cluster_id = cluster['Cluster'].values[0]
            special_day = tuple(cluster[['season', 'day']].values[0])
            cluster_weight = (df_days[(df_days['season'] == season) & (df_days['Cluster'] == cluster_id)].shape[0]) // 24
            special_days.append({'days': special_day, 'weight': cluster_weight})
            indices_to_remove.update(df_days[(df_days['season'] == season) & (df_days['Cluster'] == cluster_id)].index)
        else:
            raise ValueError(f"All days are already special days.")

    return special_days


def find_special_days(df_energy, columns=None):
    """Find special days within the representative year.

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.

    Returns
    -------
    special_days: pd.DataFrame
        DataFrame with the special days.
    """
    # Find the special days within the representative year

    df = df_energy.copy()

    def get_special_day(df, feature, rule):
        if rule == 'min':
            min_prod = df.groupby(['season', 'day'])[feature].sum().unstack().idxmin(axis=1)
            min_prod = list(min_prod.items())
            return min_prod
        else:  # rule = 'max'
            max_load = df.groupby(['season', 'day'])[feature].sum().unstack().idxmax(axis=1)
            max_load = list(max_load.items())
            return max_load

    special_days = {}

    feature = 'PV'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='min')
        special_days.update({feature: special_day})

    feature = 'Wind'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='min')
        special_days.update({feature: special_day})

    feature = 'Load'
    columns = [col for col in df_energy.columns if feature in col]
    if len(columns) > 0:
        df.loc[:, feature] = df.loc[:, columns].sum(axis=1)
        special_day = get_special_day(df, feature, rule='max')
        special_days.update({feature: special_day})

    # Format special days
    special_days = sorted([item for sublist in special_days.values() for item in sublist])
    special_days = pd.Series(special_days)
    special_days = pd.concat((special_days, pd.Series(1, index=special_days.index)), keys=['days', 'weight'], axis=1)
    special_days = special_days.set_index('days').groupby('days').first().reset_index()

    return special_days


def removed_special_days(df_energy, special_days):
    # Remove all lines in dict_info
    for special_day in special_days['days']:
        df_energy = df_energy[~((df_energy['season'] == special_day[0]) & (df_energy['day'] == special_day[1]))]

    return df_energy


def calculate_pairwise_correlation(df):
    """ Calculate correlation between all columns in a DataFrame on a row-by-row basis,
    and store the result in a new column for each pair.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the energy data.
    """
    columns = [i for i in df.columns if i not in ['season', 'day', 'hour']]
    new_columns = {}

    # Precompute means for efficiency
    means = {col: df[col].mean() for col in columns}

    for col1, col2 in combinations(columns, 2):
        corr_col_name = f"{col1}_{col2}_corr"
        new_columns[corr_col_name] = (df[col1] - means[col1]) * (df[col2] - means[col2])

    # Add all new columns at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df


def format_optim_repr_days(df_energy, name_data, folder_process_data):
    """Format the data for the optimization.

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    name_data: str
        Name of the zone.
    folder_process_data: str
        Path to save the file.
    """
    df_formatted_optim = df_energy.copy()
    # Add correlation
    df_formatted_optim = calculate_pairwise_correlation(df_formatted_optim)
    df_formatted_optim.set_index(['season', 'day', 'hour'], inplace=True)
    df_formatted_optim.index.names = [''] * 3
    # TODO: check this, currently removing zone
    # # Add header to the DataFrame with the name of the zone
    # df_formatted_optim = pd.concat([df_formatted_optim], keys=[name_data], axis=1)
    # # Invert the order of levels
    # df_formatted_optim = df_formatted_optim.swaplevel(0, 1, axis=1)

    path_data_file = os.path.join(folder_process_data, 'data_formatted_optim_{}.csv'.format(name_data))
    df_formatted_optim.to_csv(path_data_file, index=True)
    print('File saved at:', path_data_file)

    return df_formatted_optim, path_data_file


def launch_optim_repr_days(path_data_file, folder_process_data, nbr_days=3,
                           gams_model='OptimizationModel', bins_settings='settings'):
    """Launch the representative dyas optimization.

    Parameters
    ----------
    path_data_file: str
        Path to the data file.
    folder_process_data: str
        Path to save the .gms file.
    """
    if sys.platform.startswith("win"):
        path_main_file = os.path.join(os.getcwd(), f'gams\{gams_model}.gms')
        path_setting_file = os.path.join(os.getcwd(), f'gams\{bins_settings}.csv')
    else:
        path_main_file = os.path.join(os.getcwd(), f'gams/{gams_model}.gms')
        path_setting_file = os.path.join(os.getcwd(), f'gams/{bins_settings}.csv')
    path_data_file = os.path.join(os.getcwd(), path_data_file)
    print(path_main_file, path_data_file, path_data_file)

    if os.path.isfile(path_main_file) and os.path.isfile(path_data_file):
        command = ["gams", path_main_file] + ["--data {}".format(path_data_file),
                                              "--settings {}".format(path_setting_file),
                                              "--N {}".format(nbr_days)]
    else:
        raise ValueError('Gams file or data file not found')

    # Print the command
    cwd = os.path.join(os.getcwd(), folder_process_data)
    print('Launch GAMS code')
    if sys.platform.startswith("win"):  # If running on Windows
        print("Command to execute:", ' '.join(command))
        subprocess.run(' '.join(command), cwd=cwd, shell=True, stdout=subprocess.DEVNULL)
    else:  # For Linux or macOS
        subprocess.run(command, cwd=cwd, stdout=subprocess.DEVNULL)
    print('End GAMS code')

    # TODO: Check if the results exist


def parse_repr_days(folder_process_data, special_days):
    """Parse the results of the optimization.

    Parameters
    ----------
    folder_process_data: str
        Path to the folder with the results.
    special_days: pd.DataFrame
        DataFrame with the special days.

    Returns
    -------
    repr_days: pd.DataFrame
    """

    def extract_gdx(file):
        """
        Extract information as pandas DataFrame from a gdx file.

        Parameters
        ----------
        file: str
            Path to the gdx file

        Returns
        -------
        epm_result: dict
            Dictionary containing the extracted information
        """
        df = {}
        container = gt.Container(file)
        for param in container.getVariables():
            if container.data[param.name].records is not None:
                df[param.name] = container.data[param.name].records.copy()

        return df

    # Extract the results
    results_optim = extract_gdx(os.path.join(folder_process_data, 'Results.gdx'))

    # From gdx result to pandas DataFrame
    weight = results_optim['w'].copy()
    weight = weight[~np.isclose(weight['level'], 0, atol=1e-6)]

    repr_days = pd.concat((
        weight.apply(lambda x: (x['s'], x['d']), axis=1),
        weight['level']), keys=['days', 'weight'], axis=1)

    repr_days['weight'] = repr_days['weight'].round().astype(int)

    # Add special days
    repr_days = pd.concat((special_days, repr_days), axis=0, ignore_index=True)

    print('Number of days: {}'.format(repr_days.shape[0]))
    print('Total weight: {}'.format(repr_days['weight'].sum()))

    # Format the data
    repr_days['season'] = repr_days['days'].apply(lambda x: x[0])
    repr_days['day'] = repr_days['days'].apply(lambda x: x[1])
    repr_days.drop(columns=['days'], inplace=True)
    repr_days = repr_days.loc[:, ['season', 'day', 'weight']]
    repr_days = repr_days.astype({'season': int, 'day': int, 'weight': float})
    repr_days.sort_values(['season'], inplace=True)
    repr_days['daytype'] = repr_days.groupby('season').cumcount() + 1

    repr_days['season'] = repr_days['season'].apply(lambda x: f'Q{x}')

    # Update daytype naming to d1, d2...
    repr_days['daytype'] = repr_days['daytype'].apply(lambda x: f'd{x}')

    print(repr_days.groupby('season')['weight'].sum())

    return repr_days


def format_epm_phours(repr_days, folder, name_data=''):
    """Format pHours EPM like.

    Parameters
    ----------
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    folder: str
        Path to save the file.
    """
    repr_days_formatted_epm = repr_days.copy()
    repr_days_formatted_epm = repr_days_formatted_epm.set_index(['season', 'daytype'])['weight'].squeeze()
    repr_days_formatted_epm = pd.concat([repr_days_formatted_epm] * 24,
                                        keys=['t{}'.format(i) for i in range(1, 25)], names=['hour'], axis=1)

    path_file = os.path.join(folder, 'pHours_{}.csv'.format(name_data))
    repr_days_formatted_epm.to_csv(path_file)
    print('File saved at:', path_file)
    print('Number of hours: {:.0f}'.format(repr_days_formatted_epm.sum().sum() / len(repr_days_formatted_epm.columns)))


def format_epm_pvreprofile(df_energy, repr_days, folder, name_data=''):
    """Format pVREProfile EPM like

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    name_data: str
        Name of the zone.
    """
    pVREProfile = df_energy.copy()
    pVREProfile['season'] = pVREProfile['season'].apply(lambda x: f'Q{x}')

    pVREProfile = pVREProfile.set_index(['season', 'day', 'hour'])
    pVREProfile = pVREProfile[[col for col in df_energy.columns if (('PV' in col) or ('Wind' in col))]]
    pVREProfile.columns = pd.MultiIndex.from_tuples([tuple([col.split('_')[0], '_'.join(col.split('_')[1:])]) for col in pVREProfile.columns])
    if pVREProfile.columns.nlevels == 1:
        pVREProfile.columns = pd.MultiIndex.from_tuples([(col[0], name_data) for col in pVREProfile.columns])
    pVREProfile.columns.names = ['fuel', 'zone']

    t = repr_days.copy()
    t = t.set_index(['season', 'day'])
    pVREProfile = pVREProfile.unstack('hour')
    # select only the representative days
    pVREProfile = pVREProfile.loc[t.index, :]
    pVREProfile = pVREProfile.stack(level=['fuel', 'zone'])
    pVREProfile = pd.merge(pVREProfile.reset_index(), t.reset_index(), on=['season', 'day']).set_index(
        ['zone', 'season', 'daytype', 'fuel'])
    pVREProfile.drop(['day', 'weight'], axis=1, inplace=True)
    # pVREProfile = pd.concat([pVREProfile], keys=[name_data], names=['zone'], axis=0)

    pVREProfile.columns = ['t{}'.format(i + 1) for i in pVREProfile.columns]

    # Reorder index names
    pVREProfile = pVREProfile.reorder_levels(['zone', 'fuel', 'season', 'daytype'], axis=0)

    pVREProfile.to_csv(os.path.join(folder, 'pVREProfile_{}.csv'.format(name_data)), float_format='%.5f')
    print('File saved at:', os.path.join(folder, 'pVREProfile_{}.csv'.format(name_data)))


def format_epm_demandprofile(df_energy, repr_days, folder, name_data=''):
    """Format pDemandProfile EPM like

    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
    repr_days: pd.DataFrame
        DataFrame with the representative days.
    name_data: str
        Name of the zone.
    """
    pDemandProfile = df_energy.copy()
    pDemandProfile['season'] = pDemandProfile['season'].apply(lambda x: f'Q{x}')
    pDemandProfile = pDemandProfile.set_index(['season', 'day', 'hour'])
    pDemandProfile = pDemandProfile['Load'].squeeze()
    # pVREProfile.index.names = ['season', 'day', 'hour']
    t = repr_days.set_index(['season', 'day'])
    pDemandProfile = pDemandProfile.unstack('hour')
    # select only the representative days
    pDemandProfile = pDemandProfile.loc[t.index, :]
    pDemandProfile = pd.merge(pDemandProfile.reset_index(), t.reset_index(), on=['season', 'day']).set_index(
        ['season', 'daytype'])
    pDemandProfile.drop(['day', 'weight'], axis=1, inplace=True)
    pDemandProfile = pd.concat([pDemandProfile], keys=[name_data], names=['zone'], axis=0)

    pDemandProfile.columns = ['t{}'.format(i + 1) for i in pDemandProfile.columns]

    pDemandProfile.to_csv(os.path.join(folder, 'pDemandProfile_{}.csv'.format(name_data)))
    print('File saved at:', os.path.join(folder, 'pDemandProfile_{}.csv'.format(name_data)))


def cluster_data(data: pd.DataFrame, n_clusters: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Cluster the historical data into specified number of clusters.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        n_clusters (int): Number of clusters to divide the data into.

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Cluster labels and cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data.T)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=data.index, index=range(n_clusters))
    return labels, cluster_centers


def select_representative_year_real(
        data: pd.DataFrame, labels: np.ndarray, cluster_centers: pd.DataFrame
) -> pd.Series:
    """
    Select a real historical year that best represents each cluster based on the minimum distance to the cluster center.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        labels (np.ndarray): Cluster labels for each year.
        cluster_centers (pd.DataFrame): Cluster centers.

    Returns:
        pd.Series: Representative year for each cluster.
    """
    representative_years = {}
    for cluster in range(len(cluster_centers)):
        cluster_members = data.columns[labels == cluster]
        distances = [
            np.linalg.norm(data[year].values - cluster_centers.loc[cluster].values)
            for year in cluster_members
        ]
        best_year = cluster_members[np.argmin(distances)]
        representative_years[cluster] = best_year
    return pd.Series(representative_years)


def select_representative_year_synthetic(cluster_centers: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic years as the cluster centers.

    Parameters:
        cluster_centers (pd.DataFrame): Cluster centers.

    Returns:
        pd.DataFrame: Synthetic years for each cluster.
    """
    return cluster_centers.T


def assign_probabilities(labels: np.ndarray, n_clusters: int) -> pd.Series:
    """
    Calculate probabilities for each cluster based on the frequency of occurrence.

    Parameters:
        labels (np.ndarray): Cluster labels for each year.
        n_clusters (int): Number of clusters.

    Returns:
        pd.Series: Probabilities for each cluster.
    """
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return pd.Series(probabilities, index=range(n_clusters))


def run_reduced_scenarios(data: pd.DataFrame, n_clusters: int, method: str) -> pd.DataFrame:
    """
    Run the reduced scenarios algorithm to select representative years.

    Parameters:
        data (pd.DataFrame): The historical data with years as columns.
        n_clusters (int): Number of clusters to divide the data into.
        method (str): Method to select representative years.

    Returns:
        pd.DataFrame: Representative years for each cluster.
    """

    labels, cluster_centers = cluster_data(data, n_clusters)
    probabilities = assign_probabilities(labels, n_clusters)

    if method == "real":
        representatives_years = select_representative_year_real(data, labels, cluster_centers)
        representatives = data.loc[:, representatives_years]
        representatives.columns = ['{} - {:.2f}'.format(i, probabilities[k]) for k, i in representatives_years.items()]
    elif method == "synthetic":
        representatives = select_representative_year_synthetic(cluster_centers)
        representatives.columns = ['{} - {:.2f}'.format(i, probabilities[i]) for i in representatives.columns]
    else:
        raise ValueError("Invalid method")

    return representatives


def plot_uncertainty(df, df2=None, title="Uncertainty Range Plot", ylabel="Values", xlabel="Month", ymin=0, ymax=None,
                     filename=None,
                     convert_months=True):
    """
    Plots the range of values (min to max) as a transparent grey area and highlights
    the interquartile range (25th to 75th percentile) in darker grey.

    Parameters:
    - df: DataFrame with months as the index and multiple years as columns.
    - title: Title of the plot.
    - ylabel: Label for the y-axis.
    - xlabel: Label for the x-axis.
    """
    # Calculate min, max, and interquartile range
    min_vals = df.min(axis=1)
    max_vals = df.max(axis=1)
    q10_vals = df.quantile(0.10, axis=1)
    q90_vals = df.quantile(0.90, axis=1)
    q25_vals = df.quantile(0.25, axis=1)
    q75_vals = df.quantile(0.75, axis=1)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the full uncertainty range (min to max) as light grey
    ax.fill_between(df.index, min_vals, max_vals, color='grey', alpha=0.3, label='Min-Max Range')

    # Plot 10th-90th percentile in blue
    ax.fill_between(df.index, q10_vals, q90_vals, color='grey', alpha=0.5, label='10th-90th Percentile')

    # Plot the interquartile range (25th to 75th percentile) as darker grey
    ax.fill_between(df.index, q25_vals, q75_vals, color='grey', alpha=0.9, label='25th-75th Percentile')

    if df2 is not None:
        df2.plot(ax=ax)

    # Convert x-axis labels to month names if requested
    if convert_months:
        ax.set_xticks(df.index)
        ax.set_xticklabels([calendar.month_abbr[m] for m in df.index], rotation=0)

    # Add labels, title, and legend
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend(loc='upper right')

    ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    # Show the plot
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def format_dispatch_ax(ax, pd_index, day='day', season='season', display_day=True):
    # Adding the representative days and seasons
    n_rep_days = len(pd_index.get_level_values(day).unique())
    dispatch_seasons = pd_index.get_level_values(season).unique()
    total_days = len(dispatch_seasons) * n_rep_days
    y_max = ax.get_ylim()[1]

    if display_day:
        for d in range(total_days):
            x_d = 24 * d

            # Add vertical lines to separate days
            is_end_of_season = d % n_rep_days == 0
            linestyle = '-' if is_end_of_season else '--'
            ax.axvline(x=x_d, color='slategrey', linestyle=linestyle, linewidth=0.8)

            # Add day labels (d1, d2, ...)
            ax.text(
                x=x_d + 12,  # Center of the day (24 hours per day)
                y=y_max * 0.99,
                s=f'd{(d % n_rep_days) + 1}',
                ha='center',
                fontsize=7
            )

    # Add season labels
    season_x_positions = [24 * n_rep_days * s + 12 * n_rep_days for s in range(len(dispatch_seasons))]
    ax.set_xticks(season_x_positions)
    ax.set_xticklabels(dispatch_seasons, fontsize=8)
    ax.set_xlim(left=0, right=24 * total_days)
    ax.set_xlabel('Time')
    # Remove grid
    ax.grid(False)
    # Remove top spine to let days appear
    ax.spines['top'].set_visible(False)


def select_time_period(df, select_time):
    """Select a specific time period in a dataframe.

    Parameters
    ----------
    df: pd.DataFrame
        Columns contain season and day
    select_time: dict
        For each key, specifies a subset of the dataframe

    Returns
    -------
    pd.DataFrame: Dataframe with the selected time period
    str: String with the selected time period
    """
    temp = df.copy()
    for k, i in select_time.items():
        temp = temp.loc[temp.index.get_level_values(k).isin(i), :]
    return temp


def create_season_day_index() -> pd.Series:
    """Create a MultiIndex with all combinations of seasons and days.

    Returns:
    -------
    pd.Series: Series with a MultiIndex of seasons and days.
    """

    # Create the levels for the MultiIndex
    months = np.arange(1, 13)  # Months 1-12
    days = np.arange(1, 32)  # Days 1-31

    # Generate all combinations of months and days
    month_day_combinations = [(month, day) for month in months for day in days]

    # Filter invalid days (e.g., February 30, April 31)
    valid_combinations = [
        (month, day)
        for month, day in month_day_combinations
        if day <= pd.Timestamp(f"2025-{month:02d}-01").days_in_month
    ]

    # Create the MultiIndex
    multi_index = pd.MultiIndex.from_tuples(valid_combinations, names=["season", "day"])

    # Create the Series with 1 as the value everywhere
    series = pd.Series(1, index=multi_index)

    return series


def plot_dispatch(df, day_level='daytype', season_level='season', title=None):
    fig, ax = plt.subplots()

    df.plot(ax=ax)

    format_dispatch_ax(ax, df.index)

    ax.set_xlabel('Hours')
    ax.set_ylim(bottom=0)
    if title is not None:
        ax.set_title(title)
    plt.show()


def make_heatmap(df, tech):
    df_energy = df.copy()
    daily_df = df_energy.groupby(['season', 'day'], as_index=False).sum(numeric_only=True).drop(columns='hour')
    daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)
    tmp = daily_df.sort_values(['season', 'day']).copy()

    heatmap_data = tmp[[col for col in tmp.columns if tech in col]]
    heatmap_data.index = tmp['season-day']

    # Get tick positions where the season changes
    season_labels = tmp['season'].values
    x_positions = []
    x_labels = []
    for i in range(1, len(season_labels)):
        if season_labels[i] != season_labels[i - 1]:
            x_positions.append(i)
            x_labels.append(season_labels[i])
    # Add the first season manually
    x_positions = [0] + x_positions
    x_labels = [season_labels[0]] + x_labels

    # Plot
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data.T, cmap='YlGnBu', xticklabels=False)

    # Custom x-ticks with season only
    plt.xticks(ticks=x_positions, labels=x_labels, rotation=0)
    plt.xlabel("Season")
    plt.ylabel("Zone")
    plt.title(f"Heatmap of Daily {tech} Generation (Grouped by Season)")
    plt.tight_layout()
    plt.show()


def make_boxplot(df, tech):
    df_energy = df.copy()
    daily_df = df_energy.groupby(['season', 'day'], as_index=False).sum(numeric_only=True).drop(columns='hour')
    daily_df['season-day'] = daily_df["season"].astype(str) + " - " + daily_df["day"].astype(str)
    tmp = daily_df.sort_values(['season', 'day']).copy()

    melted = tmp.melt(id_vars=['season', 'day'], value_vars=[col for col in tmp.columns if tech in col],
                      var_name='zone', value_name='daily_sum')

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=melted, x="zone", y="daily_sum", hue="season")
    plt.xticks(rotation=45)
    plt.title(f"Distribution of Daily {tech} Generation by Season and Zone")
    plt.tight_layout()
    plt.show()


def plot_vre_repdays(input_file, vre_profile, pHours, season_colors, countries=None,
                   fontsize_legend=8, alpha_background=0.1, min_alpha=0.3, max_alpha=1):
    """
    Plot time series of renewable generation (e.g., PV, Wind) for all days and highlight representative days.

    This function displays hourly renewable generation profiles for all days (as background lines)
    and overlays the representative days used in the Poncelet algorithm, with line opacity
    proportional to their weight in the optimization.

    Parameters
    ----------
    input_file : pd.DataFrame
        Hourly renewable generation data with a multi-index (season, day, hour) and multi-index columns (fuel, zone).

    vre_profile : pd.DataFrame
        Output from the representative day selection, formatted with representative generation profiles.

    pHours : pd.DataFrame
        Hourly weights associated with each representative day, used to scale the line opacity.

    season_colors : dict
        Dictionary mapping each season to a color.

    countries : list, optional
        List of zones to include. If None, all available zones are used.

    fontsize_legend : int, default 8
        Font size for the legend.

    alpha_background : float, default 0.1
        Transparency for the background lines representing all days.

    min_alpha : float, default 0.3
        Minimum transparency for the representative day lines (lowest weight).

    max_alpha : float, default 1
        Maximum opacity for the representative day lines (highest weight).

    Notes
    -----
    - One subplot is generated per fuel and season.
    - Representative days are plotted with darker lines when their weight in the optimization is higher.
    - All background days are plotted in a lighter color for context.
    """

    input_file = input_file.copy()
    input_file.index = input_file.index.set_levels(
        ['Q' + str(s) for s in input_file.index.levels[0]],
        level='season'
    )
    input_file.index = input_file.index.set_levels(
        ['d' + str(s) for s in input_file.index.levels[1]],
        level='day'
    )

    # Remove correlation columns
    columns = [c for c in input_file.columns if 'corr' not in c]
    input_file = input_file[columns]

    # Reconstruct MultiIndex if needed
    if input_file.columns.nlevels == 1:
        input_file.columns = pd.MultiIndex.from_tuples(
            [tuple([col.split('_')[0], '_'.join(col.split('_')[1:])]) for col in input_file.columns])
        input_file.columns.names = ['fuel', 'zone']

    if countries is None:
        countries = input_file.columns.get_level_values('zone').unique()

    input_file = input_file.loc[:, input_file.columns.get_level_values('zone').isin(countries)]

    fuels = input_file.columns.get_level_values('fuel').unique()
    seasons = input_file.index.get_level_values('season').unique()

    df_rep = vre_profile.loc[vre_profile.index.get_level_values('zone').isin(countries), :]
    df_rep.columns = df_rep.columns.astype(str).str.extract(r't(\d+)')[0].astype(int) - 1
    df_rep = df_rep.stack()
    df_rep.index.names = ['zone', 'fuel', 'season', 'day', 'hour']
    df_rep = df_rep.unstack(['zone', 'fuel'])

    df_hours = pHours.copy()
    df_hours.columns = df_hours.columns.astype(str).str.extract(r't(\d+)')[0].astype(int) - 1
    df_hours = df_hours.stack()
    df_hours.index.names = ['season', 'day', 'hour']

    # Get min/max weights for scaling alpha
    wmin, wmax = df_hours.min(), df_hours.max()

    def scale_alpha(w):
        if wmax == wmin:
            return (min_alpha + max_alpha) / 2
        return min_alpha + (w - wmin) / (wmax - wmin) * (max_alpha - min_alpha)


    fig, axs = plt.subplots(len(fuels), len(seasons), figsize=(5 * len(seasons), 3.5 * len(fuels)), sharey=True, sharex=True)

    # Always use 2D indexing for axs
    if len(fuels) == 1:
        axs = axs[np.newaxis, :]
    if len(seasons) == 1:
        axs = axs[:, np.newaxis]

    for i, fuel in enumerate(fuels):
        subset_days = input_file.loc[:, input_file.columns.get_level_values('fuel') == fuel]
        subset_repdays = df_rep.loc[:, df_rep.columns.get_level_values('fuel') == fuel]

        for j, season in enumerate(seasons):
            ax = axs[i][j]
            season_data = subset_days.loc[season]
            season_repdays = subset_repdays.loc[season]

            for day in season_data.index.get_level_values('day').unique():
                day_data = season_data.loc[day]
                ax.plot(
                    day_data.index.get_level_values('hour'),
                    day_data.sum(axis=1),
                    color=season_colors.get(season, 'orange'),
                    alpha=alpha_background
                )

            for repday in season_repdays.index.get_level_values('day').unique():
                day_data = season_repdays.loc[repday]
                weight = df_hours.xs(season, level='season').xs(repday, level='day').mean()
                label = f'{season}-{repday} ({int(weight)})'

                alpha = scale_alpha(weight)
                ax.plot(
                    day_data.index.get_level_values('hour'),
                    day_data.sum(axis=1),
                    color=season_colors.get(season, 'grey'),
                    alpha=alpha,
                    linewidth = 2,
                    label=label,
                    zorder=2
                )

            ax.set_title(f"{fuel} - {season}")
            ax.legend(loc='upper right', fontsize=fontsize_legend, frameon=False)
            if j == 0:
                ax.set_ylabel("Generation")
            ax.grid(True)

    axs[-1][-1].set_xlabel("Hour of day")
    plt.tight_layout()
    plt.show()


def run_smoothing_reservoir(config):
    """
    Smooths the energy dispatch of a hydro reservoir by solving a linear optimization problem.

    The function models a hydro power plant with a reservoir and optimizes its hourly dispatch
    to meet inflow and storage constraints while minimizing variability in dispatch across time
    and across seasons. It ensures the system operates within physical limits such as reservoir capacity,
    minimum seasonal storage, and seasonal balance in dispatch.

    Args:
        config (dict): Dictionary containing model parameters. Required keys are:
            - 'hours' (int): Total number of time steps (typically hours).
            - 'inflow' (np.ndarray): Array of inflows into the reservoir [GWh] of shape (hours,).
            - 'reservoir_size' (float): Maximum storage capacity of the reservoir [GWh].
            - 'reservoir_min' (float): Minimum storage level as a fraction of reservoir_size
              (e.g., 0.2 for 20%). Enforced only during specified hours.
            - 'hours_reservoir_min_constraints' (np.ndarray): Array of time indices (ints)
              where the minimum storage level must be enforced.

    Returns:
        res (scipy.optimize.OptimizeResult): Result object from `scipy.optimize.linprog` containing:
            - res.x: Optimal values of the decision variables (inflow, outflow, storage, dispatch, abs_diff, D_j, U_j).
            - res.fun: Value of the objective function at the optimum (total smoothing cost).
            - res.success: Boolean indicating whether optimization succeeded.
            - res.message: Solver status message.
    """

    # Parameters
    hours = config['hours']
    reservoir_max = config['reservoir_size']  # in GWh
    min_storage = config['reservoir_min'] * reservoir_max
    seasonal_hours = config['hours_reservoir_min_constraints']  # must be a numpy array containing the t for which the reservoir constraint should be enforced
    inflow = config['inflow']  # should be of shape (hours,)

    # Variables: [i_t, o_t, s_t, d_t, u_t, D_j, U_j] for each hour, plus D1-D4, plus U1-U3
    n = hours
    n_var = 5 * n + 4 + 3  # hourly vars + seasonal dispatch sums + abs diffs

    # Objective: minimize sum of U_j
    c = np.zeros(n_var)
    c[5 * n + 4:] = 1  # U1, U2, U3
    c[4: 5 * n: 5] = 1  # u_t terms

    # Bounds
    bounds = [(0, None), (0, None), (0, reservoir_max), (0, None), (0, None)] * n
    bounds += [(0, None)] * 4  # D1-D4
    bounds += [(0, None)] * 3  # U1-U3

    # Equality constraints: storage + dispatch + seasonal sums
    A_eq = lil_matrix((2 * n + 4, n_var))
    b_eq = np.zeros(2 * n + 4)

    for t in range(n):
        # Storage equation: s_t - s_{t-1} - i_t + o_t = 0
        if t > 0:
            A_eq[t - 1, 5 * t + 2] = 1  # s_t
            A_eq[t - 1, 5 * t + 0] = -1  # i_t
            A_eq[t - 1, 5 * t + 1] = 1  # o_t
            A_eq[t - 1, 5 * (t - 1) + 2] = -1  # s_{t-1}

        # Dispatch equation: d_t + i_t - o_t = inflow_t
        A_eq[n - 1 + t, 5 * t + 3] = 1  # d_t
        A_eq[n - 1 + t, 5 * t + 0] = 1  # i_t
        A_eq[n - 1 + t, 5 * t + 1] = -1  # o_t
        b_eq[n - 1 + t] = inflow[t]

    # Equation to ensure storage level is the same at beginning and end
    A_eq[2 * n - 1, 2] = 1
    A_eq[2 * n - 1, 5 * (n - 1) + 2] = -1

    # Seasonal sum constraints: D_j = sum(d_t in season j)
    season_length = n // 4
    for j in range(4):
        for t in range(j * season_length, (j + 1) * season_length):
            A_eq[2 * n + j, 5 * t + 3] = 1  # coefficients for each d_t in the season
        A_eq[2 * n + j, 5 * n + j] = -1  # subtract D_j

    # Inequality constraints
    row_max = 2 * (n - 1) + len(seasonal_hours) + 6
    A_ub = lil_matrix((row_max, n_var))
    b_ub = np.zeros(row_max)

    row = 0

    # Absolute value for difference in outcomes
    for t in range(1, n):
        # u_t >= |d_t - d_{t-1}|
        A_ub[row, 5 * t + 4] = -1
        A_ub[row, 5 * t + 3] = 1
        A_ub[row, 5 * (t - 1) + 3] = -1
        row += 1

        A_ub[row, 5 * t + 4] = -1
        A_ub[row, 5 * t + 3] = -1
        A_ub[row, 5 * (t - 1) + 3] = 1
        row += 1

    # Seasonal storage constraint: s_t >= min → -s_t <= -min
    for t in seasonal_hours:
        A_ub[row, 5 * t + 2] = -1
        b_ub[row] = -min_storage
        row += 1

    # U_j >= |D_{j+1} - D_j|
    for j in range(3):
        # D_{j+1} - D_j - U_j <= 0
        A_ub[row, 5 * n + j + 1] = 1
        A_ub[row, 5 * n + j] = -1
        A_ub[row, 5 * n + 4 + j] = -1
        row += 1

        # D_{j} - D_{j+1} - U_j <= 0
        A_ub[row, 5 * n + j + 1] = -1
        A_ub[row, 5 * n + j] = 1
        A_ub[row, 5 * n + 4 + j] = -1
        row += 1

    # Solve
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub[:row],
                   A_eq=A_eq.tocsr(), b_eq=b_eq,
                   bounds=bounds, method='highs')

    # Output result status
    print(res.success, res.fun)
    return res, A_eq, b_eq

