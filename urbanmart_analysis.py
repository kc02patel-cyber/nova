import pandas as pd

def compute_daily_sales(csv_path, start_date=None, end_date=None, date_col='date'):
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    df['line_revenue'] = df['quantity'] * df['unit_price'] - df['discount_applied']
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    grouped = (
        df.groupby(date_col)
        .agg(daily_revenue=('line_revenue','sum'),
             daily_transactions=('transaction_id','nunique'))
        .sort_index()
    )
    full_index = pd.date_range(grouped.index.min(), grouped.index.max(), freq='D')
    grouped = grouped.reindex(full_index, fill_value=0.0)
    grouped.index.name='date'
    grouped['cumulative_revenue'] = grouped['daily_revenue'].cumsum()
    grouped['moving_avg_7d'] = grouped['daily_revenue'].rolling(7, min_periods=1).mean()
    grouped['day_of_week'] = grouped.index.day_name()
    return grouped

if __name__=='__main__':
    df_daily = compute_daily_sales('urbanmart_sales.csv')
    print(df_daily)
    df_daily.to_csv('daily_sales.csv')
