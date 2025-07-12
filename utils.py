from typing import List
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import json
import os

# global variables for folder paths
EOM_json_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\EOM\json"
EOM_html_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\EOM\html"
regular_json_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\regular\json"
regular_html_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\regular\html"
weekly_json_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\weekly\json"
weekly_html_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\weekly\html"
quarterly_json_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\quarterly\json"
quarterly_html_folder = os.getcwd() + r"\raw_data\CboeSettlementValues\quarterly\html"

# all_settlemnet_values file path
all_settlement_values_file = os.getcwd() + r"\all_settlement_values.json"

# functions block for settlement value fill and corresponding data analysis
def convert_html_to_json(html_json_file: str, output_json_file: str, is_regular: bool):
    """
    void function converts the settlement values in html string to json
    :param html_json_file: json containing the html string settlement values
    :param output_json_file: json output
    :param is_regular: whether is regular option
    :return: None
    """
    json_data = []
    regular_table_name_suffix = "_Settlement Values"

    with open(html_json_file, 'r', encoding='utf-8') as f:
        data_file = json.load(f)

    soup = BeautifulSoup(data_file['data'], 'html.parser')
    tables = soup.find_all('table')
    table_names = soup.find_all('h4')

    for table, table_name in zip(tables, table_names):

        table_date_str = table_name.get_text(strip=True)

        if is_regular:
            table_date_str = table_date_str[:-len(regular_table_name_suffix)]
        dt = datetime.strptime(table_date_str, "%B %Y")
        formatted_table_date = dt.strftime("%Y-%m")

        rows = table.find('tbody').find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            if cols:
                if not is_regular:
                    date_str = cols[2].get_text(strip=True)
                    try:
                        dt = datetime.strptime(date_str, "%m/%d/%Y")
                    except ValueError:
                        try:
                            dt = datetime.strptime(date_str, "%m/%d/%y")
                        except ValueError:
                            formatted_date = date_str
                        else:
                            formatted_date = dt.strftime("%Y-%m-%d")
                    else:
                        formatted_date = dt.strftime("%Y-%m-%d")

                    json_data.append(
                        {
                            "description": cols[0].get_text(strip=True),
                            "trading_symbol": cols[1].get_text(strip=True),
                            "expiration_date": formatted_date,
                            "settlement_symbol": cols[3].get_text(strip=True),
                            "settlement_value": cols[4].get_text(strip=True),
                            "year_month": formatted_table_date
                        }
                    )
                else:
                    parts = cols[0].get_text(strip=True).split(" (")
                    json_data.append(
                        {
                            "description": parts[0],
                            "settlement_symbol": parts[1].strip("):"),
                            "settlement_value": cols[1].get_text(strip=True),
                            "year_month": formatted_table_date
                        }
                    )

    json_dict = {
        'data': json_data,
    }

    with open(output_json_file, "w") as f:
        json.dump(json_dict, f, indent=2)

    return

def convert_all_htmls_to_jsons():
    """
    Void function to convert all htmls json files to json files settlement values
    :return: None
    """
    for root, dirs, files in os.walk(EOM_html_folder):
        for file in files:
            input_file = EOM_html_folder + '\\' + file
            output_file = EOM_json_folder + '\\' + file[:4] + '.json'
            convert_html_to_json(input_file, output_file, False)

    for root, dirs, files in os.walk(weekly_html_folder):
        for file in files:
            input_file = weekly_html_folder + '\\' + file
            output_file = weekly_json_folder + '\\' + file[:4] + '.json'
            convert_html_to_json(input_file, output_file, False)

    for root, dirs, files in os.walk(regular_html_folder):
        for file in files:
            input_file = regular_html_folder + '\\' + file
            output_file = regular_json_folder + '\\' + file[:4] + '.json'
            convert_html_to_json(input_file, output_file, True)

    for root, dirs, files in os.walk(quarterly_html_folder):
        for file in files:
            input_file = quarterly_html_folder + '\\' + file
            output_file = quarterly_json_folder + '\\' + file[:4] + '.json'
            convert_html_to_json(input_file, output_file, False)

def generate_all_settlement_values_json():
    """
    void function generates a json file with all SPX 500 option settlement values
    :return: None
    """
    # so we can extract all SPX 500 option settlement values and combine in only one json file
    all_settlement_values = {}
    output_json_file = os.getcwd() + r"\all_settlement_values.json"

    EOM_dict = {}
    quarterly_dict = {}
    weekly_dict = {}
    regular_dict = {}

    # ('S&P 500 CLOSE', 'SPX')
    for root, dires, files in os.walk(EOM_json_folder):
        for file in files:
            spx = []
            with open(EOM_json_folder + '\\' + file, 'r') as f:
                sv = json.load(f)['data']
            for i in sv:
                if (i['description'], i['settlement_symbol']) == ('S&P 500 CLOSE', 'SPX'):
                    spx.append(i)
            EOM_dict[file[:4]] = spx

    # ('S&P 500', 'SPX')
    for root, dires, files in os.walk(quarterly_json_folder):
        for file in files:
            spx = []
            with open(quarterly_json_folder + '\\' + file, 'r') as f:
                sv = json.load(f)['data']
            for i in sv:
                if (i['description'], i['settlement_symbol']) == ('S&P 500', 'SPX'):
                    spx.append(i)
            quarterly_dict[file[:4]] = spx

    # ('S&P 500', 'SET'), ('S&P 500', 'SPX'), ('S&P 500 CLOSE', 'SPX')
    for root, dires, files in os.walk(weekly_json_folder):
        for file in files:
            spx = []
            with open(weekly_json_folder + '\\' + file, 'r') as f:
                sv = json.load(f)['data']
            for i in sv:
                if (
                    (i['description'], i['settlement_symbol']) == ('S&P 500', 'SET') or 
                    (i['description'], i['settlement_symbol']) == ('S&P 500', 'SPX') or 
                    (i['description'], i['settlement_symbol']) == ('S&P 500 CLOSE', 'SPX')
                ):
                    spx.append(i)
            weekly_dict[file[:4]] = spx

    # ('S&P 500', 'SET'), ('S&P 500 (SET)', 'SET'), ('S&P 500 PM Settled Options', 'SPX'), 
    # ('S&P 500 PM Settled Options', 'SPXPM'), ('S&P 500 PM-Settled', 'SPX')
    for root, dires, files in os.walk(regular_json_folder):
        for file in files:
            spx = []
            with open(regular_json_folder + '\\' + file, 'r') as f:
                sv = json.load(f)['data']
            for i in sv:
                if (
                    (i['description'], i['settlement_symbol']) == ('S&P 500', 'SET') or 
                    (i['description'], i['settlement_symbol']) == ('S&P 500 (SET)', 'SET') or 
                    (i['description'], i['settlement_symbol']) == ('S&P 500 PM Settled Options', 'SPX') or 
                    (i['description'], i['settlement_symbol']) == ('S&P 500 PM Settled Options', 'SPXPM') or
                    (i['description'], i['settlement_symbol']) == ('S&P 500 PM-Settled', 'SPX')
                ):
                    spx.append(i)
            regular_dict[file[:4]] = spx


    all_settlement_values['EOM'] = EOM_dict
    all_settlement_values['quarterly'] = quarterly_dict
    all_settlement_values['weekly'] = weekly_dict
    all_settlement_values['regular'] = regular_dict

    with open(output_json_file, "w") as f:
        json.dump(all_settlement_values, f, indent=2)

def generate_filled_OSI_df(OSI_df: pd.DataFrame, security_close_prices: dict[str, float]) -> pd.DataFrame:
    """
    function to fill in the settlement values, settlement symbols and contract starting code
    :param OSI_df: the pandas dataframe about to be filled in
    :param security_close_prices: the security close prices serving as the unknow PM close price reference
    :return: filled pandas dataframe with filled settlement value, symbol and contract starting code
    """
    OSI_df = OSI_df.fillna({'expiry_indicator': 'r'})
    OSI_df['settlement_value'] = float(0)
    OSI_df['settlement_symbol'] = 'missing'
    OSI_df['starting_code'] = 'missing'

    SPX_list = ['weekly', 'EOM', 'quarterly']

    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    for index, row in tqdm(OSI_df.iterrows(), total=len(OSI_df)):

        is_filled = False

        expiration_date = row['exdate']
        expiry_indicator = row['expiry_indicator']
        year = expiration_date[:4]
        year_month = expiration_date[:7]
        code = row['symbol'].split(' ')[0]

        OSI_df.at[index, 'starting_code'] = code

        # for SPXPM
        if code == 'SPXPM':

            if (
                expiration_date == '2010-12-31' or 
                expiration_date == '2011-03-31'
            ):
                OSI_df.at[index, 'settlement_value'] = float(security_close_prices[expiration_date])
                OSI_df.at[index, 'settlement_symbol'] = 'SPX'
            
            else:
                continue

        # for SPXQ
        if code == 'SPXQ':

            if expiration_date == '2011-06-30':

                OSI_df.at[index, 'settlement_value'] = float(security_close_prices[expiration_date])
                OSI_df.at[index, 'settlement_symbol'] = 'SPX'
            
            else:
                for type in settlement_values.keys():
                    if is_filled:
                        break
                    if type in SPX_list:
                        if year in settlement_values[type]:
                            for i in settlement_values[type][year]:
                                if is_filled:
                                    break
                                if (i['expiration_date'] == expiration_date):
                                    try:
                                        OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                        OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                    except:
                                        continue
                                    else:
                                        is_filled = True

        # for SPX
        if code == 'SPX' and (expiry_indicator == 'm' or expiry_indicator == 'w'):

            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if (i['expiration_date'] == expiration_date):
                        try:
                            OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            break

        if code == 'SPX' and expiry_indicator == 'r':

            if year in settlement_values['regular']:
                for i in settlement_values['regular'][year]:
                    if 'expiration_date' in i:
                        if (
                            i['expiration_date'] == expiration_date and
                            i['settlement_symbol'] == 'SET'
                        ):
                            try:
                                OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                            except:
                                continue
                            else:
                                break
                    else:
                        if (
                            i['year_month'] == year_month and
                            i['settlement_symbol'] == 'SET'
                        ):
                            try:
                                OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                            except:
                                continue
                            else:
                                break

        # for SPXW
        if code == 'SPXW' and expiry_indicator == 'm':
 
            for type in settlement_values.keys():
                if is_filled:
                    break
                if type in SPX_list:
                    if year in settlement_values[type]:
                        for i in settlement_values[type][year]:
                            if is_filled:
                                break
                            if (i['expiration_date'] == expiration_date):
                                try:
                                    OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                    OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                except:
                                    continue
                                else:
                                    is_filled = True

        if code == 'SPXW' and expiry_indicator == 'r':

            # search in weekly first
            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if is_filled:
                        break
                    if (i['expiration_date'] == expiration_date):
                        try:
                            OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            is_filled = True

            # if not found in weekly, search in regular PM
            if not is_filled:
                if year in settlement_values['regular']:
                    for i in settlement_values['regular'][year]:
                        if is_filled:
                            break
                        if 'expiration_date' in i:
                            if (
                                i['expiration_date'] == expiration_date and
                                (i['settlement_symbol'] == 'SPX' or i['settlement_symbol'] == 'SPXPM')
                            ):
                                try:
                                    OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                    OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                except:
                                    continue
                                else:
                                    is_filled = True
                        else:
                            if (
                                i['year_month'] == year_month and
                                (i['settlement_symbol'] == 'SPX' or i['settlement_symbol'] == 'SPXPM')
                            ):
                                try:
                                    OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                    OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                except:
                                    continue
                                else:
                                    is_filled = True

            # if not found yet, extract from security prices
            if not is_filled:
                OSI_df.at[index, 'settlement_value'] = float(security_close_prices[expiration_date])
                OSI_df.at[index, 'settlement_symbol'] = 'SPXPM'

        if code == 'SPXW' and expiry_indicator == 'w':
            

            # 2018-12-05 shall be removed, 2017-10-25 CBOE missing
            if expiration_date == '2017-10-25':
                OSI_df.at[index, 'settlement_value'] = float(security_close_prices[expiration_date])
                OSI_df.at[index, 'settlement_symbol'] = 'SPX'
                # skip the searching code below
                continue

                # continue to next row of the dataset, expiration date 2018-12-04 shall be removed
            if expiration_date == '2018-12-05':
                # though not existing, settlement_symbol should be SPX
                OSI_df.at[index, 'settlement_symbol'] = 'SPX'
                # skip the searching code below
                continue

            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if is_filled:
                        break
                    if (i['expiration_date'] == expiration_date):
                        try:
                            OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            is_filled = True
            
            # specifically for 2022-10-21
            if not is_filled:
                if year in settlement_values['regular']:
                    for i in settlement_values['regular'][year]:
                        if is_filled:
                            break
                        if 'expiration_date' in i:
                            if (
                                i['expiration_date'] == expiration_date and
                                (i['settlement_symbol'] == 'SPX' or i['settlement_symbol'] == 'SPXPM')
                            ):
                                try:
                                    OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                    OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                except:
                                    continue
                                else:
                                    is_filled = True
                        else:
                            if (
                                i['year_month'] == year_month and
                                (i['settlement_symbol'] == 'SPX' or i['settlement_symbol'] == 'SPXPM')
                            ):
                                try:
                                    OSI_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                    OSI_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                                except:
                                    continue
                                else:
                                    is_filled = True

    return OSI_df

def generate_filled_old_df(old_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to fill in the information of old options data frame
    Notice we shall consider the overlap between weekly Friday and quarterly
    :param old_df: the pandas data frame about to be filled
    :return: the filled pandas data frame
    """
    old_df = old_df.fillna({'expiry_indicator': 'r'})
    old_df['settlement_value'] = float(0)
    old_df['settlement_symbol'] = 'missing'
    old_df['starting_code'] = 'missing'

    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    for index, row in tqdm(old_df.iterrows(), total=len(old_df)):

        expiration_date = row['exdate']
        year = expiration_date[:4]
        year_month = expiration_date[:7]

        if '.' in row['symbol']:
            code = row['symbol'].split('.')[0]
        else:
            code = row['symbol'].split(' ')[0]

        old_df.at[index, 'starting_code'] = code

        # three special dates
        # since the three special cases do not have e.g. JXA trading symbol (theirs are like SZQ)
        # so we use the expiration week number to match
        if expiration_date == '2007-03-30':
            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if (
                        i['year_month'] == year_month and 
                        i['expiration_date'] == 'FIVE' and 
                        i['settlement_symbol'] == 'SET'
                    ):
                        try:
                            old_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            old_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            break

        elif expiration_date == '2007-06-29':
            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if (
                        i['year_month'] == year_month and 
                        i['expiration_date'] == 'FIVE' and 
                        i['settlement_symbol'] == 'SET'
                    ):
                        try:
                            old_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            old_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            break
        
        elif expiration_date == '2007-09-28':
            if year in settlement_values['weekly']:
                for i in settlement_values['weekly'][year]:
                    if (
                        i['year_month'] == year_month and 
                        i['expiration_date'] == 'FOUR' and 
                        i['settlement_symbol'] == 'SET'
                    ):
                        try:
                            old_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                            old_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                        except:
                            continue
                        else:
                            break
        
        else:

            # for regular 3rd Friday options
            if (
                is_third_friday(expiration_date) or 
                is_third_friday(expiration_date, -1) or 
                is_third_friday(expiration_date, 1)
            ):
                if year in settlement_values['regular']:
                    for i in settlement_values['regular'][year]:
                        if (
                            i['year_month'] == year_month and
                            i['settlement_symbol'] == 'SET'
                        ):
                            try:
                                old_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                old_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                            except:
                                continue
                            else:
                                break

            # for the rest weeklies
            elif (
                is_friday(expiration_date) or 
                is_friday(expiration_date, -1) or 
                is_friday(expiration_date, 1)
            ):  
                if year in settlement_values['weekly']:
                    for i in settlement_values['weekly'][year]:
                        if (
                            i['year_month'] == year_month and 
                            i['trading_symbol'] == code and 
                            i['settlement_symbol'] == 'SET'
                        ):
                            try:
                                old_df.at[index, 'settlement_value'] = float(i['settlement_value'])
                                old_df.at[index, 'settlement_symbol'] = i['settlement_symbol']
                            except:
                                continue
                            else:
                                break

            # for quarterly we skip
            else:
                continue
    
    return old_df

def is_friday(date: str, change: int = 0) -> bool:
    """
    Check if the given date is a Friday.
    :param date: Date in 'YYYY-MM-DD' format
    :param change: next x days of the current date
    :return: True if the date is a Friday, False otherwise
    """
    return (pd.to_datetime(date) + pd.Timedelta(days=change)).day_name() == 'Friday'

def is_third_friday(date: str, change: int = 0) -> bool:
    """
    Check if the given date is the third Friday of the month.
    :param date: Date in 'YYYY-MM-DD' format
    :param change: After x days
    :return: True if the date is the third Friday, False otherwise
    """
    date = pd.to_datetime(date) + pd.Timedelta(days=change)
    if date.weekday() != 4:  # 4 = Friday
        return False
    # Get all Fridays of that month
    fridays = pd.date_range(start=date.replace(day=1), 
                            end=date.replace(day=28) + pd.offsets.MonthEnd(0), 
                            freq='W-FRI')
    return date == fridays[2]

def is_month_end(date: str, change: int = 0) -> bool:
    """
    Function to check whehther the date is the end of month or close to end of month
    :param date: the date to be checked
    :param cahnge: the next x days of the current date
    :return: True if the date is or is very close to the end of the month
    """
    date = pd.to_datetime(date)
    return (date + pd.Timedelta(days=change)).is_month_end

def is_quarter_end(date: str, change: int = 0) -> bool:
    """
    Function to check whehther the date is the end of quarter or close to end of quarter
    :param date: the date to be checked
    :param cahnge: the next x days of the current date
    :return: True if the date is or is very close to the end of the quarter
    """
    date = pd.to_datetime(date)
    return (date + pd.Timedelta(days=change)).is_quarter_end

def create_weekly_dates() -> List[str]:
    """
    Create the list to contain all dates in the weekly settlement values
    Notice in this list, there are week numbers like ONE, TWO
    :return: List containing all dates in the weekly settlement values
    """
    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    weekly_dates = []
    for year, data in settlement_values['weekly'].items():
        for i in data:
            weekly_dates.append(i['expiration_date'])

    return weekly_dates

def create_regular_dates() -> List[str]:
    """
    Create the list to contain existing expiration dates in the regular settlement values
    :return: List containing all existing expiration dates in regular settlement values
    """
    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    regular_dates = []
    for year, data in settlement_values['regular'].items():
        for i in data:
            if 'expiration_date' in i:
                regular_dates.append(i['expiration_date'])

    return regular_dates

def create_EOM_dates() -> List[str]:
    """
    Create the list to contain existing expiration dates in the EOM settlement values
    :return: List containing all existing expiration dates in EOM settlement values
    """
    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    EOM_dates = []
    for year, data in settlement_values['EOM'].items():
        for i in data:
            EOM_dates.append(i['expiration_date'])

    return EOM_dates

def create_quarterly_dates() -> List[str]:
    """
    Create the list to contain existing expiration dates in the quarterly settlement values
    :return: List containing all existing expiration dates in quarterly settlement values
    """
    input_file = all_settlement_values_file
    with open(input_file, 'r') as f:
        settlement_values = json.load(f)

    quarterly_dates = []
    for year, data in settlement_values['quarterly'].items():
        for i in data:
            quarterly_dates.append(i['expiration_date'])

    return quarterly_dates

# functions block for dataset splitting and adding feature
def find_atm_option(group, **kwargs):
    """
    Function to find the row of most likely ATM option
    Internal function used in pandas group apply
    """
    idx = (group['moneyness'] - 1).abs().idxmin() # find index of minimum distance to 1
    return group.loc[idx]