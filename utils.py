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


API_TOKEN = '38bb40c2e0090463d92457a7bb87af45fdbba28b'


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
        response = requests.get(url, headers=headers)

        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            # Append the data along with location details
            for timestamp, values in data['data'].items():
                row = {
                    'latitude': lat,
                    'longitude': lon,
                    'timestamp': pd.to_datetime(int(timestamp) / 1000, unit='s'),
                    #'local_time': pd.to_datetime(int(local_time) / 1000, unit='s'),
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
    return df


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
    for year in range(start_year, end_year):

        start_date = '{}-{}'.format(year, start_day)
        end_date = '{}-{}'.format(year, end_day)

        # Call the function to get data
        data = get_renewable_data(power_type, locations, start_date, end_date, turbine=turbine)

        if data.empty:
            print("No data for year {}".format(year))
            continue
        else:
            print("Getting data {} for year {}".format(power_type, year))
            data.set_index(['latitude', 'longitude'], inplace=True)
            data['local_time'] = pd.to_datetime(data['local_time'])
            data['season'] = data['local_time'].dt.month
            data['day'] = data['local_time'].dt.day
            data['hour'] = data['local_time'].dt.hour
            data['time_only'] = data['local_time'].dt.strftime('%m-%d %H:%M:%S')
            data.set_index(['time_only', 'season', 'day', 'hour'], inplace=True, append=True)
            data = data.loc[:, power_type]

            results.update({year: data})

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


def format_data_energy(filenames):
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
    for key, filename in filenames.items():
        # Extract from the data results 
        results = pd.read_csv(filename, header=[0], index_col=[0, 1, 2, 3, 4, 5])

        repr_year = find_representative_year(results)

        # Format the data
        df = results.loc[:, repr_year]
        df = df.reset_index()
        df = df.loc[:, ['season', 'day', 'hour', repr_year]].rename(columns={repr_year: key})
        
        df_energy.update({key: df})
        
    df_energy = pd.merge(df_energy['PV'], df_energy['Wind'], on=['season', 'day', 'hour'])

    print('Annual capacity factor (%):', df_energy[['PV', 'Wind']].mean())
    
    return df_energy


def find_special_days(df_energy):
    """Find special days within the representative year.
    
    Parameters
    ----------
    df_energy: pd.DataFrame
        DataFrame with the energy data.
        
    Returns
    -------
    special_days: pd.DataFrame
        DataFrame with the special days.
    df_energy: pd.DataFrame
        DataFrame without the special days.
    """
    # Find the special days within the representative year
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
        
    # Remove all lines in dict_info
    for special_day in special_days.values():
        df_energy = df_energy[~((df_energy['season'] == special_day[0]) & (df_energy['day'] == special_day[1]))]
        
    # Format special days
    special_days = sorted([item for sublist in special_days.values() for item in sublist])
    special_days = pd.Series(special_days)
    special_days = pd.concat((special_days, pd.Series(1, index=special_days.index)), keys=['days', 'weight'], axis=1)
    special_days = special_days.set_index('days').groupby('days').sum().reset_index()
    
    return special_days, df_energy


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
        for col2 in columns[i+1:]:
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
    
    # Add correlation
    df_formatted_optim = calculate_pairwise_correlation(df_energy)
    df_formatted_optim.set_index(['season', 'day', 'hour'], inplace=True)
    df_formatted_optim.index.names = [''] * 3
    # Add header to the DataFrame with the name of the zone
    df_formatted_optim = pd.concat([df_formatted_optim], keys=[name_data], axis=1)
    # Invert the order of levels
    df_formatted_optim = df_formatted_optim.swaplevel(0, 1, axis=1)
    
    path_data_file = os.path.join(folder_process_data, 'data_formatted_optim_{}.csv'.format(name_data))
    df_formatted_optim.to_csv(path_data_file, index=True)
    print('File saved at:', path_data_file)
    
    return df_formatted_optim, path_data_file


def launch_optim_repr_days(path_data_file, folder_process_data):
    """Launch the representative dyas optimization.
    
    Parameters
    ----------
    path_data_file: str
        Path to the data file.
    folder_process_data: str
        Path to save the .gms file.
    """
    
    path_main_file = os.path.join(os.getcwd(),'gams/OptimizationModel.gms')
    path_setting_file = os.path.join(os.getcwd(),'gams/settings.csv')
    path_data_file = os.path.join(os.getcwd(), path_data_file)
    nbr_days = 3


    if os.path.isfile(path_main_file) and os.path.isfile(path_data_file):
        command = ["gams", path_main_file] + ["--data {}".format(path_data_file),
                                            "--settings {}".format(path_setting_file),
                                            "--N {}".format(nbr_days)]
    else:
        raise ValueError('Gams file or data file not found')

    # Print the command
    print("Command to execute:", command)
    
    cwd = os.path.join(os.getcwd(), folder_process_data)

    subprocess.run(command, cwd=cwd, stdout=subprocess.DEVNULL)
    
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
    weight = weight[weight['level'] != 0]

    repr_days = pd.concat((
        weight.apply(lambda x: (x['s'], x['d']), axis=1),
        weight['level']), keys=['days', 'weight'], axis=1)

    # Add special days
    repr_days = pd.concat((special_days, repr_days), axis=0, ignore_index=True)

    print('Number of days:', repr_days.shape[0])
    print('Total weight:', repr_days['weight'].sum())
    
    # Format the data
    repr_days['season'] = repr_days['days'].apply(lambda x: x[0])
    repr_days['day'] = repr_days['days'].apply(lambda x: x[1])
    repr_days.drop(columns=['days'], inplace=True)
    repr_days = repr_days.loc[:, ['season', 'day', 'weight']]
    repr_days = repr_days.astype({'season': int, 'day': int, 'weight': int})
    repr_days.sort_values(['season', 'weight'], inplace=True)
    repr_days['daytype'] = repr_days.groupby('season').cumcount() + 1
    
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
    repr_days_formatted_epm = repr_days.set_index(['season', 'daytype']).squeeze()
    repr_days_formatted_epm = pd.concat([repr_days_formatted_epm] * 24, keys=['t{:02d}'.format(i) for i in range(1, 25)], names=['hour'], axis=1)
    
    path_file = os.path.join(folder, 'pHours_{}.csv'.format(name_data))
    repr_days_formatted_epm.to_csv(path_file)
    print('File saved at:', path_file)
    print('Number of hours: {:.0f}'.format(repr_days_formatted_epm.sum().sum()))
    
    
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
    
    pVREProfile = df_energy[[i for i in ['PV', 'Wind']]]
    pVREProfile.index.names = ['season', 'day', 'hour']
    pVREProfile.stack().unstack('hour')
    pVREProfile.columns.names = ['Power']
    t = repr_days.set_index(['season', 'day'])
    temp = pVREProfile.unstack('hour')
    # select only the representative days
    temp = temp.loc[t.index, :]
    temp = temp.stack('Power')
    temp = pd.merge(temp.reset_index(), t.reset_index(), on=['season', 'day']).set_index(['season', 'daytype', 'Power'])
    temp.drop(['day', 'weight'], axis=1, inplace=True)
    temp = pd.concat([temp], keys=[name_data], names=['zone'], axis=0)
    
    temp.columns = ['t{:02d}'.format(i + 1) for i in temp.columns]
    
    temp.to_csv(os.path.join(folder, 'pVREProfile_{}.csv'.format(name_data)))
    print('File saved at:', os.path.join(folder, 'pVREProfile_{}.csv'.format(name_data)))
    
