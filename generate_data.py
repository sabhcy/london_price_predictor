"""Generate a realistic sample London housing dataset (~15k sales, 2019-2026)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

# borough: (price per sqft GBP 2019, lat, lon, inner?)
B = {
 "Kensington and Chelsea":(1450,51.502,-0.194,1),"Westminster":(1350,51.497,-0.137,1),
 "City of London":(1200,51.515,-0.092,1),"Camden":(1050,51.529,-0.125,1),
 "Hammersmith and Fulham":(900,51.492,-0.223,1),"Islington":(880,51.538,-0.103,1),
 "Richmond upon Thames":(760,51.461,-0.303,0),"Wandsworth":(780,51.457,-0.192,1),
 "Hackney":(720,51.545,-0.055,1),"Lambeth":(700,51.460,-0.121,1),
 "Southwark":(690,51.503,-0.080,1),"Tower Hamlets":(640,51.515,-0.034,1),
 "Kingston upon Thames":(560,51.412,-0.301,0),"Haringey":(610,51.590,-0.111,0),
 "Barnet":(570,51.625,-0.152,0),"Merton":(590,51.410,-0.188,0),
 "Ealing":(560,51.513,-0.308,0),"Brent":(540,51.558,-0.282,0),
 "Hounslow":(470,51.468,-0.361,0),"Harrow":(480,51.580,-0.334,0),
 "Greenwich":(510,51.483,0.006,0),"Lewisham":(500,51.445,-0.021,0),
 "Waltham Forest":(510,51.588,-0.012,0),"Hillingdon":(440,51.534,-0.452,0),
 "Enfield":(440,51.652,-0.081,0),"Bromley":(470,51.402,0.015,0),
 "Sutton":(420,51.361,-0.194,0),"Croydon":(410,51.372,-0.100,0),
 "Redbridge":(440,51.559,0.074,0),"Newham":(450,51.525,0.035,0),
 "Havering":(380,51.577,0.212,0),"Bexley":(370,51.455,0.150,0),
 "Barking and Dagenham":(340,51.554,0.134,0),
}
types = ["Flat","Terraced","Semi-Detached","Detached"]
n = 15000
rows = []
names = list(B)
w = np.array([0.6 if B[b][3] else 1.0 for b in names]); w /= w.sum()
for _ in range(n):
    b = rng.choice(names, p=w); psf, lat, lon, inner = B[b]
    pt = rng.choice(types, p=[.62,.2,.12,.06] if inner else [.42,.3,.18,.10])
    beds = {"Flat":rng.choice([0,1,2,3],p=[.08,.42,.4,.1]),
            "Terraced":rng.choice([2,3,4,5],p=[.25,.45,.22,.08]),
            "Semi-Detached":rng.choice([2,3,4,5],p=[.1,.5,.3,.1]),
            "Detached":rng.choice([3,4,5,6],p=[.2,.4,.3,.1])}[pt]
    sqft = max(300, rng.normal(420 + beds*260 + (150 if pt=="Detached" else 0), 120))
    baths = max(1, min(beds, int(round(beds*0.6 + rng.normal(0,.4)))))
    tube = abs(rng.gamma(2, 0.35 if inner else 0.7))
    tenure = "Leasehold" if pt=="Flat" or rng.random()<.08 else "Freehold"
    new = rng.random() < (.18 if pt=="Flat" else .04)
    year_built = int(rng.integers(2015,2026)) if new else int(rng.choice([1880,1910,1930,1960,1985,2005]) + rng.integers(0,20))
    epc = rng.choice(list("ABCDEFG"), p=[.03,.25,.32,.28,.09,.02,.01]) if not new else rng.choice(["A","B"])
    garden = pt!="Flat" or rng.random()<.15
    parking = rng.random() < (.2 if inner else .55)
    d = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 2800)))
    t = (d.year-2019) + d.dayofyear/365
    market = 1 + 0.035*t - 0.012*max(0,t-3.8)**1.5   # growth then 2023 rate-shock flattening
    type_mult = {"Flat":0.92,"Terraced":1.0,"Semi-Detached":1.03,"Detached":1.12}[pt]
    epc_mult = {"A":1.05,"B":1.04,"C":1.02,"D":1,"E":.97,"F":.94,"G":.91}[epc]
    price = (psf*sqft*market*type_mult*epc_mult
             *(1-0.035*tube)*(1.06 if new else 1)*(1.03 if garden else 1)
             *(1.04 if parking and inner else 1.01 if parking else 1)
             *(0.96 if tenure=="Leasehold" and pt!="Flat" else 1)
             *rng.lognormal(0,0.11))
    rows.append(dict(sale_date=d.date(), borough=b, property_type=pt, tenure=tenure,
        new_build=new, bedrooms=beds, bathrooms=baths, floor_area_sqft=round(sqft),
        year_built=year_built, epc_rating=epc, has_garden=garden, has_parking=parking,
        dist_to_tube_km=round(tube,2),
        latitude=round(lat+rng.normal(0,.012),5), longitude=round(lon+rng.normal(0,.018),5),
        price_gbp=int(round(price,-3))))
df = pd.DataFrame(rows).sort_values("sale_date")
df.to_csv("data/london_housing.csv", index=False)
print(df.shape); print(df.price_gbp.describe())
print(df.groupby("borough").price_gbp.median().sort_values().tail(3))
