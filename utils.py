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

logging.basicConfig(level=logging.WARNING)  # Configure logging level
logger = logging.getLogger(__name__)

API_TOKEN = '38bb40c2e0090463d92457a7bb87af45fdbba28b'
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
    requests_made = 0
    minute_start_time = time.time()

    all_data = []

    # Iterate over locations
    for lat, lon in locations:
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
        requests_made += 1

        # If we hit 6 requests within a minute, we wait for the remaining time of that minute
        if requests_made >= 6:
            print('Waiting for a minute to not hit the API rate limit...')
            elapsed_time = time.time() - minute_start_time
            if elapsed_time < 60:
                sleep_time = 60 - elapsed_time
                print(f"Hit rate limit. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)

            # Reset the counter and timer
            requests_made = 0
            minute_start_time = time.time()

    # Convert the collected data into a DataFrame
    df = pd.DataFrame(all_data)
    df.rename(columns={'electricity': power_type}, inplace=True)
    return df, requests_made


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

        # If we hit 50 requests within an hour, we wait for the remaining time of that minute before continuing the requests
        if requests_tot >= 49:
            print('Waiting for a minute to not hit the API rate limit...')
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
            repr_year = find_representative_year(df)
            print('Representative year {}'.format(repr_year))
            # Format the data
            df = df.loc[:, repr_year]
            df = df.reset_index()
            df['zone'] = list(zip(df['latitude'], df['longitude']))  # Convert lat/lon to tuples
            df['zone'] = df['zone'].map(locations)
        elif reading == 'standard':
            df = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3])
            repr_year = find_representative_year(df)
            print('Representative year {}'.format(repr_year))
            df = df.reset_index()
        else:
            raise ValueError('Unknown reading. Only implemented for: renewable_ninja, standard.')

        df = df.loc[:, ['zone', 'season', 'day', 'hour', repr_year]].rename(columns={repr_year: key})
        df = df.sort_values(by=['zone', 'season', 'day', 'hour'], ascending=True).reset_index(drop=True)

        # If 2/29, remove it
        if len(df.season.unique()) == 12:  # season expressed as months
            df = df[~((df['season'] == 2) & (df['day'] == 29))]

        df_energy.update({key: df})

    keys_to_merge = ['PV', 'Wind', 'Load', 'ROR']
    keys_to_merge = [i for i in keys_to_merge if i in df_energy.keys()]

    # Dynamically merge all DataFrames in df_energy based on the specified keys
    df_energy = reduce(
        lambda left, right: pd.merge(left, right, on=['zone', 'season', 'day', 'hour']),
        (df_energy[k] for k in keys_to_merge))

    if len(df_energy.season.unique()) == 12:  # only when seasons are the months, we remove the 29th of February
        df_energy = df_energy[~((df_energy['season'] == 2) & (df_energy['day'] == 29))]

    if df_energy.isna().any().any():
        print('Warning: NaN values in the DataFrame')

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
    Perform KMeans clustering on energy data to identify representative days.

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

        # second special day is based on maximum peak demand across all zones
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
    Select extreme days (e.g., minimum PV, maximum Load) to ensure proper representation of special conditions.

    If an extreme day is already selected, it continues looking for the next most extreme day until a unique one is found.

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
    if columns is None:
        columns = [i for i in df_energy.columns if i not in ['season', 'day', 'hour']]

    special_days = {}
    for column in columns:
        # Remove the day with the minimum production
        if column in ['Wind', 'PV', 'ROR']:
            min_prod = df_energy.groupby(['season', 'day'])[column].sum().unstack().idxmin(axis=1)
            min_prod = list(min_prod.items())

            special_days.update({column: min_prod})
        elif column in ['Load']:
            max_load = df_energy.groupby(['season', 'day'])[column].sum().unstack().idxmax(axis=1)
            max_load = list(max_load.items())

            special_days.update({column: max_load})
        else:
            raise ValueError('Unknown column. Only implemented for: Wind, PV, ROR, Load.')

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

    # Iterate through all pairs of columns
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            # Calculate the correlation for each row
            corr_col_name = f"{col1}{col2}corr"
            df[corr_col_name] = (df[col1] - df[col1].mean()) * (df[col2] - df[col2].mean())

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
    pVREProfile.columns = pd.MultiIndex.from_tuples([tuple(col.split('_')) for col in pVREProfile.columns])
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
