# Make sure you're in the project folder with venv active, then run:
# python dat.py

import pandas as pd
from app import create_app
from api.extensions import db
from api.models.models import Order, Warehouse
from datetime import datetime

app = create_app()
with app.app_context():
    # Seed warehouses first
    warehouses = [
        Warehouse(warehouse_id='WH_North', name='North Warehouse', city='Delhi',    capacity=600),
        Warehouse(warehouse_id='WH_South', name='South Warehouse', city='Chennai',  capacity=550),
        Warehouse(warehouse_id='WH_East',  name='East Warehouse',  city='Kolkata',  capacity=500),
        Warehouse(warehouse_id='WH_West',  name='West Warehouse',  city='Mumbai',   capacity=700),
    ]
    for w in warehouses:
        db.session.merge(w)
    db.session.commit()
    print('Warehouses seeded')

    # Import orders from CSV
    df = pd.read_csv('ecommerce_data.csv')
    df = df.head(500)  # start with 500 rows
    count = 0
    for _, row in df.iterrows():
        o = Order(
            order_id               = row['order_id'],
            warehouse_id           = row['warehouse'],
            city                   = row['city'],
            distance_km            = int(row['distance_km']),
            order_value            = float(row['order_value']),
            order_items            = int(row['order_items']),
            warehouse_load         = int(row['warehouse_load']),
            promised_delivery_days = int(row['promised_delivery_days']),
            actual_delivery_days   = int(row['actual_delivery_days']),
            is_delayed             = bool(row['is_delayed']),
            delay_days             = int(row['delay_days']),
            is_returned            = bool(row['is_returned']),
            past_delays            = int(row['past_delays']),
            delivery_cost          = float(row['delivery_cost']),
            return_cost            = float(row['return_cost']),
            status                 = 'delivered',
            order_date             = datetime.strptime(str(row['order_date'])[:10], '%Y-%m-%d'),
        )
        db.session.merge(o)
        count += 1
    db.session.commit()
    print(f'Imported {count} orders successfully!')