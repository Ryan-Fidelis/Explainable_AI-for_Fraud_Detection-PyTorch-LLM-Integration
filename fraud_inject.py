import numpy as np
import pandas as pd
import random
from dataload import load_data as ld

def fraud():
    # 1. DATA LOADING
    dfs = ld()
    df_emp = dfs[0]
    df_dep = dfs[1]
    df_merc = dfs[2]
    df_exp = dfs[3]

    # 2. SYNTHETIC FRAUD INJECTION
    def inject_frauds(df_expenses, df_employees):
        """
        Injects 4 synthetic fraud patterns into the expense transactions DataFrame.
        """
        df_exp = df_expenses.copy()
        
        if 'Fraud' not in df_exp.columns:
            df_exp['Fraud'] = 0
        else:
            df_exp['Fraud'] = df_exp['Fraud'].fillna(0).astype(int)

        # ---------------------------------------------------------
        # Pattern 1: Expense Splitting (Smurfing)
        # ---------------------------------------------------------
        for j in range(10):
            mask_high_value = (df_exp['Amount'] > 5000) & (df_exp['Fraud'] == 0)
            if mask_high_value.any():
                smurf_idx = np.random.choice(df_exp[mask_high_value].index)
                row_smurf = df_exp.loc[smurf_idx].copy()
                
                df_exp = df_exp.drop(smurf_idx)
                
                num_splits = 3
                split_amount = row_smurf['Amount'] / num_splits
                smurf_rows = []
                base_id = df_exp['ID'].max() if not df_exp.empty else 0
                
                for i in range(num_splits):
                    new_row = row_smurf.copy()
                    new_row['Amount'] = split_amount
                    new_row['ID'] = base_id + 1 + i
                    new_row['Fraud'] = 1
                    new_row['transaction_hour'] = min(23, int(new_row['transaction_hour']) + i)
                    smurf_rows.append(new_row)
                    
                df_exp = pd.concat([df_exp, pd.DataFrame(smurf_rows)], ignore_index=True)

        # ---------------------------------------------------------
        # Pattern 2: Out-of-Hours Anomaly (Late Night / Weekend)
        # ---------------------------------------------------------
        mask_legit_weekday = (df_exp['Fraud'] == 0) & (df_exp['is_weekend'] == 0)
        out_of_hours_idx = df_exp[mask_legit_weekday].sample(frac=0.02, random_state=42).index
        
        for idx in out_of_hours_idx:
            if random.choice([True, False]):
                # Late night (01:00 to 05:00)
                df_exp.loc[idx, 'transaction_hour'] = random.randint(1, 5)
            else:
                # Sunday
                df_exp.loc[idx, 'transaction_day_of_week'] = 6
                df_exp.loc[idx, 'is_weekend'] = 1
            df_exp.loc[idx, 'Fraud'] = 1

        # ---------------------------------------------------------
        # Pattern 3: Expense Inflation (Policy Violation via Z-Score)
        # ---------------------------------------------------------
        mask_legit = (df_exp['Fraud'] == 0)
        inflation_idx = df_exp[mask_legit].sample(frac=0.01, random_state=123).index
        
        fatores = np.random.uniform(4.0, 8.0, size=len(inflation_idx))
        df_exp.loc[inflation_idx, 'Amount'] *= fatores
        df_exp.loc[inflation_idx, 'Fraud'] = 1

        # ---------------------------------------------------------
        # Pattern 4: Shell Merchant Collusion
        # ---------------------------------------------------------
        fake_merchant_id = 999999
        random_emp_id = df_employees['ID'].sample(1).values[0]
        
        shell_rows = []
        base_id = df_exp['ID'].max() if not df_exp.empty else 0
        
        for i in range(15):
            shell_rows.append({
                'ID': base_id + 100 + i,
                'Employee_ID': random_emp_id,
                'Merchant_ID': fake_merchant_id,
                'Amount': random.choice([1000.0, 2000.0, 3000.0, 5000.0]),
                'Status': 'Closed',
                'Fraud': 1,
                'transaction_hour': random.randint(10, 16),
                'transaction_day_of_week': random.randint(0, 4),
                'is_weekend': 0
            })
            
        df_exp = pd.concat([df_exp, pd.DataFrame(shell_rows)], ignore_index=True)
        return df_exp

    # 3. DEPARTMENT AGGREGATION METRICS
    def department_amount():
        merged_df = df_exp.merge(df_emp, left_on='Employee_ID', right_on='ID', suffixes=('_expense', '_employee'))
        full_df = merged_df.merge(df_dep, left_on='Department_ID', right_on='ID', suffixes=('_employee', '_department'))
        
        result = full_df.groupby('Department_ID').agg(
            total_gasto=('Amount', 'sum'),
            media_por_transacao=('Amount', 'mean'),
            desvio_padrao_amount=('Amount', 'std'),
            qtd_funcionarios=('Employee_ID', 'nunique')
        ).reset_index()

        result['media_por_funcionario'] = result['total_gasto'] / result['qtd_funcionarios']
        result.drop(columns=['qtd_funcionarios'], inplace=True)
        return result

    # 4. FEATURE ENGINEERING & PREPROCESSING
    def department_amount_z_score():
        # Merge Employees and Departments
        merged_df = df_exp.merge(
            df_emp, 
            left_on='Employee_ID', 
            right_on='ID', 
            suffixes=('_expense', '_employee')
        )
        full_df = merged_df.merge(
            df_dep, 
            left_on='Department_ID', 
            right_on='ID', 
            suffixes=('_employee', '_department')
        )
        
        # Merge Merchant Category
        full_df = full_df.merge(
            df_merc[['ID', 'Category']], 
            left_on='Merchant_ID', 
            right_on='ID', 
            how='left',
            suffixes=('', '_merchant')
        )
        full_df['Category'] = full_df['Category'].fillna('Shell_Company')
        
        all_df = full_df.merge(
            department_amount(),
            on='Department_ID'
        )
        
        # Calculate amount Z-Score
        all_df['amount_z_score'] = (all_df['Amount'] - all_df['media_por_transacao']) / all_df['desvio_padrao_amount']
        
        column_remove = [
            'Employee_ID', 'ID_expense', 'Department_ID', 'Amount', 'total_gasto', 
            'Status_expense', 'Merchant_ID', 'Cost_Center_Code', 'Name_department', 
            'Expense_Limit', 'Status_employee', 'CPF', 'Name_employee', 'ID_employee', 
            'is_weekend', 'media_por_funcionario', 'ID_department', 'ID', 'ID_merchant',
            'media_por_transacao', 'desvio_padrao_amount'
        ]
        
        columns_ex = [col for col in column_remove if col in all_df.columns]
        all_df.drop(columns_ex, axis=1, inplace=True)

        # Cyclical transformations for time features
        all_df['hour_sin'] = np.sin(2 * np.pi * all_df['transaction_hour'] / 24.0)
        all_df['hour_cos'] = np.cos(2 * np.pi * all_df['transaction_hour'] / 24.0)
        all_df['day_sin'] = np.sin(2 * np.pi * all_df['transaction_day_of_week'] / 7.0)
        all_df['day_cos'] = np.cos(2 * np.pi * all_df['transaction_day_of_week'] / 7.0)
        
        all_df.drop(['transaction_hour', 'transaction_day_of_week'], axis=1, inplace=True)

        all_df = pd.get_dummies(all_df, columns=['Role_Level', 'Category'], dtype=float)

        return all_df

    # Execute pipeline
    df_exp = inject_frauds(df_exp, df_emp)
    last_df = department_amount_z_score()
    
    return last_df