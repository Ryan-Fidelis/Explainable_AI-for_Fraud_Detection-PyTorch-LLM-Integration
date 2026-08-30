import pandas as pd

def load_data():

    employee = ['ID', 'Department_ID','CPF', 'Name', 'Status', 'Role_Level']
    department = ['ID', 'Name', 'Cost_Center_Code', 'Expense_Limit']
    merchant = ['ID', 'CNPJ', 'Name', 'Category']
    expense = ['ID', 'Employee_ID', 'Merchant_ID', 'Amount', 'Status', 'Fraud', 'transaction_hour', 'transaction_day_of_week', 'is_weekend']

    df_emp = pd.read_csv('Employee.csv', sep=';', header=None, names=employee)
    df_dep = pd.read_csv('Department.csv', sep=';', header=None, names=department)
    df_merc = pd.read_csv('Merchant.csv', sep=';', header=None, names=merchant)
    df_exp = pd.read_csv('Expense_Transaction.csv', sep=';', header=None, names=expense)
    
    return df_emp, df_dep, df_merc, df_exp