from datetime import datetime

def is_nan(value):
    """
    Returns true if value is NaN, false otherwise
    Parameters:
         value (Object): An object to test
    """

    return value != value

def whats(thing) :
    """
    Prints the type of object passed in
    Parameters:
        thing (Object): The object for which the type needs to be printed
    """

    print(type(thing))

def parse_billingenddate_datetime(transactionDateTime):
    """
     Parses a billing period end date in the format 'YYYY-MM-DD'
     Parameters:
        transactionDateTime (String): Formatted as 'YYYY-MM-DD'
    """

    # Return the parsed value
    return datetime.strptime(transactionDateTime, '%Y-%m-%d')

def find_month_from_billing_end_date(date):
    """
     Takes a date object and tell you what month this data pertains to (eg. 24th Jan = Jan, 02nd Jan = Dec of previous year)
     Parameters:
        date(datetime): The datetime object
    """

    if(date.day >= 15):
        return date.month
    else:
        if(date.month == 1):
            return 12
        else:
            return date.month - 1

def find_year_from_billing_end_date(date):
    """
     Takes a date object and tell you what year this data pertains to (eg. 24th Jan = same year, 02nd Jan = previous year)
     Parameters:
        date(datetime): The datetime object
    """

    if(date.month == 1 and date.day < 15):
        return date.year - 1
    else:
        return date.year

def find_month_from_fati_reporting_month(month_text):
    """
     Take the text in the reporting month column and returns the month it represents
     Parameters:
        month_text(string): The reporting month text
    """

    swticher = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

    return swticher.get(month_text[0:3])


def find_year_from_fati_reporting_month(month_text):
    """
     Take the text in the reporting month column and returns the month it represents
     Parameters:
        month_text(string): The reporting month text
    """

    return int(month_text[4:8])

def find_customer_stage_for_month_year(customerid, month, year, customer_stage_df):
    """
     Given the customer Id, month, and year, finds the customer stage in the given df
     Parameters:
        customerid(int): The customer Id
        month(int): The month number
        year(year): The year number
        customer_stage_df(dataframe): The customer stage dataframe

    """

    row = customer_stage_df.loc[(customer_stage_df['#############'] == customerid) & (customer_stage_df['Month'] == month) & (customer_stage_df['Year'] == year)]

    if (row.size == 0):
        return 'STAGE NOT FOUND'
    else:
        return row['CUSTOMERSTAGE'].iloc[0]

def get_power_user(customerstage):
    """
     Given the customer stage in text, return 1 if power user or 0 if not
     Parameters:
        customerstage(string): The stage of the customer
    """

    if(customerstage == '3. Power'):
        return 1
    else:
        return 0