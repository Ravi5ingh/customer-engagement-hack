from Util import *
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import *
from sklearn.metrics import confusion_matrix

def generate_fati():
    """
    From the FATI table, gets the records for the 'usable' customers and generate a new table
    """
    customerids = pd.read_csv('#############\CustomerIds.csv')
    fati = pd.read_csv('#############\AllUserFATI.csv')
    fatiUsable = fati.loc[fati['#############'].isin(customerids['#############'])]
    fatiUsable.to_csv('#############\FATI.csv', index=False)
    print('Done generating FATI')

def generate_customer_stage():
    """
    From the CMBillingPeriod table, gets the records for the 'usable' customers and generate a new table
    """
    customerids = pd.read_csv('#############\CustomerIds.csv')
    cmbillingperiodChunks = pd.read_csv('#############\AllUserCMBillingPeriods.csv', chunksize=10000)

    chunkIndex = 0
    cmBillingPeriods = pd.DataFrame()
    for cmChunk in cmbillingperiodChunks:
        if (chunkIndex == 0):
            cmBillingPeriods = pd.DataFrame(columns=['#############', 'BILLINGENDDATE', 'CUSTOMERSTAGE'])

        prunedChunk = cmChunk.loc[cmChunk['#############'].isin(customerids['#############'])]
        cmBillingPeriods = cmBillingPeriods.append(prunedChunk.loc[:, ['#############', 'BILLINGENDDATE', 'CUSTOMERSTAGE']])
        chunkIndex += 1
        print('Processed ' + str(chunkIndex) + ' chunks. Current size of accumulated DF: ' + str(cmBillingPeriods.shape[0]))

    cmBillingPeriods.to_csv('#############\CustomerStage.csv', index=False)
    print('Done generating customer stage')

def categorize_months_customer_stage():
    """
    For each record in CMBillingPeriod, assume it represents a month based on the end date
    """
    customerStages = pd.read_csv('#############\CustomerStage.csv')
    parsedDates = customerStages['BILLINGENDDATE'].apply(parse_billingenddate_datetime)
    customerStages['Month'] = parsedDates.apply(lambda x: find_month_from_billing_end_date(x))
    customerStages['Year'] = parsedDates.apply(lambda x: find_year_from_billing_end_date(x))

    customerStages.to_csv('#############\\1_CustomerStage_MonthCat.csv', index=False)
    print('Done adding month and year columns to Customer Stage table')

def categorize_months_fati():
    """
    For each record in FATI, get the month, and year it pertains to
    """
    fati = pd.read_csv('#############\FATI.csv')
    reportingMonthColumn = fati['ReportingMonthText']
    fati['Month'] = reportingMonthColumn.apply(lambda x: find_month_from_fati_reporting_month(x))
    fati['Year'] = reportingMonthColumn.apply(lambda x: int(x[4:8]))

    fati.to_csv('#############\\2_FATI_MonthCat.csv', index=False)
    print('Done adding month and year columns to FATI table')

def add_columns_for_categories():
    """
    For the categorical columns, do some one-hot-encoding
    """

    fatijoined = pd.read_csv('#############\\3_FATI_Join_CS.csv')

    fatijoined = pd.concat(
        [fatijoined,
         pd.get_dummies(fatijoined['FacilityTypeDesc'], prefix='Is')],
        axis=1
    )

    fatijoined.to_csv('#############\\4_Data_OneHotEncoded.csv', index=False)

def normalize_columns():

    fati = pd.read_csv('#############\\4_Data_OneHotEncoded.csv')

    fati = fati[(fati['CurrBalance'] >= -20000) & (fati['CurrBalance'] <= 20000)]
    fati = fati[(fati['MinBalance'] >= -20000) & (fati['MinBalance'] <= 20000)]
    fati = fati[(fati['MaxBalance'] >= -25000) & (fati['MaxBalance'] <= 25000)]
    fati = fati[(fati['AvgBalance'] >= -20000) & (fati['AvgBalance'] <= 20000)]
    fati = fati[(fati['CreditTurnover'] >= 0) & (fati['CreditTurnover'] <= 30000)]
    fati = fati[(fati['DebitTurnover'] >= 0) & (fati['DebitTurnover'] <= 20000)]
    # fati = fati[(fati['RejectedPayments'] >= -20000) & (fati['RejectedPayments'] <= 20000)]
    # fati = fati[(fati['MaxExcess'] >= -20000) & (fati['MaxExcess'] <= 20000)]

    fati['CurrBalance'] = (fati['CurrBalance'] + abs(fati['CurrBalance'].min())) / fati['CurrBalance'].max()
    fati['MinBalance'] = (fati['MinBalance'] + abs(fati['MinBalance'].min())) / fati['MinBalance'].max()
    fati['MaxBalance'] = (fati['MaxBalance'] + abs(fati['MaxBalance'].min())) / fati['MaxBalance'].max()
    fati['AvgBalance'] = (fati['CurrBalance'] + abs(fati['CurrBalance'].min())) / fati['CurrBalance'].max()
    fati['CreditTurnover'] = (fati['CreditTurnover'] + abs(fati['CreditTurnover'].min())) / fati['CreditTurnover'].max()
    fati['DebitTurnover'] = (fati['DebitTurnover'] + abs(fati['DebitTurnover'].min())) / fati['DebitTurnover'].max()
    fati['RejectedPayments'] = (fati['RejectedPayments'] + abs(fati['RejectedPayments'].min())) / fati['RejectedPayments'].max()
    fati['MaxExcess'] = (fati['MaxExcess'] + abs(fati['MaxExcess'].min())) / fati['MaxExcess'].max()

    fati.to_csv('#############\\5_Data_Normalized.csv',index=False)

    print('Done Normalizing Columns')


def label_power_user_records():

    fati = pd.read_csv('#############\\5_Data_Normalized.csv')

    fati['IsPowerUser'] = fati['CustomerStage'].apply(lambda x: get_power_user(x))

    fati.to_csv('#############\\6_Data_OutputNormalized.csv', index=False)

    print('Done labeling power user records')

def remove_unnecessary_columns():

    fati = pd.read_csv('#############\\6_Data_OutputNormalized.csv')

    pcaDf = pd.DataFrame(columns= ['Is_Credit Card', 'Is_Current Account', 'CurrBalance', 'MinBalance', 'MaxBalance', 'AvgBalance', 'CreditTurnover', 'DebitTurnover', 'RejectedPayments', 'MaxExcess', 'IsPowerUser'])

    pcaDf['Is_Credit Card'] = fati['Is_Credit Card']
    pcaDf['Is_Current Account'] = fati['Is_Current Account']
    pcaDf['CurrBalance'] = fati['CurrBalance']
    pcaDf['MinBalance'] = fati['MinBalance']
    pcaDf['MaxBalance'] = fati['MaxBalance']
    pcaDf['AvgBalance'] = fati['AvgBalance']
    pcaDf['CreditTurnover'] = fati['CreditTurnover']
    pcaDf['DebitTurnover'] = fati['DebitTurnover']
    pcaDf['RejectedPayments'] = fati['RejectedPayments']
    pcaDf['MaxExcess'] = fati['MaxExcess']
    pcaDf['IsPowerUser'] = fati['IsPowerUser']

    pcaDf.to_csv('#############\\7_Data_PcaReady.csv', index=False)

    print('Done removing un-necessary columns')

def show_pca():
    """
    Plot the results of principal component analysis and show the plot
    """

    fati = pd.read_csv('#############\\7_Data_PcaReady.csv')

    X = fati[['Is_Credit Card', 'Is_Current Account', 'CurrBalance', 'MinBalance', 'MaxBalance', 'AvgBalance', 'CreditTurnover', 'DebitTurnover', 'RejectedPayments', 'MaxExcess']]

    pca = PCA(n_components=2)

    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, fati[['IsPowerUser']]], axis=1)

    finalDf.head()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [0, 1]
    colors = ['g', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['IsPowerUser'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()

    plt.show()

def run_random_forest():

    fati = pd.read_csv('#############\\7_Data_PcaReady.csv')

    X = fati[['Is_Credit Card', 'Is_Current Account', 'CurrBalance', 'MinBalance', 'MaxBalance', 'AvgBalance', 'CreditTurnover', 'DebitTurnover', 'RejectedPayments', 'MaxExcess']]
    Y = fati['IsPowerUser']

    classifier = RandomForestClassifier(n_estimators=100)

    pipeline = Pipeline(
        [
            ('predict', classifier)
        ])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=27)

    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict_proba(x_test)[:, 1]
    y_pred = y_pred >= 0.6

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    print('Done Running Random Forest')








