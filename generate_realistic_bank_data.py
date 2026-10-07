#!/usr/bin/env python3
"""
REALISTIC synthetic retail-bank dataset (all data is fictional).

What makes it look like a real core-banking extract
  * 10-digit customer IDs (CIF), unique 12-digit account numbers
    (4-digit branch code + 2-digit product code + 6-digit serial), 14-digit loan numbers
  * Region-appropriate Indian names (South / West / North / East / Muslim / Christian / Sikh)
  * Real Indian localities with their PIN codes, branches with IFSC + MICR codes
  * Valid formats: 10-digit mobiles on real series, PAN AAAAA9999A, unique emails
  * Loans with a correct EMI formula, outstanding principal, overdue days, NPA status
  * Transactions with UPI/NEFT/IMPS/ATM/POS narrations, salary credits, EMI debits and
    a running balance_after that reconciles to the account balance

Set INJECT_ERRORS = False for a clean bank database, or True (default) to add deliberate

"""
import os
import sys

import numpy as np
import pandas as pd

import oracledb


# ------------------------------------------------------------------ config
SEED = 42
N_CUSTOMERS = 100_000        # total customer rows (incl. duplicates when INJECT_ERRORS)
DUP_RATE = 0.015
EXTRA_ACCT_RATIO = 0.30      # extra accounts (FD / savings) on top of 1 primary account each
N_LOANS = 30_000
N_TXN_RANDOM = 250_000       # UPI/ATM/POS/... ; salary + EMI transactions are added on top
N_HISTORY = 8_000
INJECT_ERRORS = True
BANK_CODE = "BHNB"           # fictional bank (IFSC prefix)
BANK_MICR = "045"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(SEED)
END = pd.Timestamp("2026-09-23")
START = END - pd.Timedelta(days=179)
DAY = pd.Timedelta(days=1)
SOURCES = np.array(["CBS", "MOBILE_APP", "BRANCH_PORTAL", "LOAN_ORIGINATION"])
SRC_P_CUST = [0.45, 0.25, 0.20, 0.10]
SRC_P_LOAN = [0.30, 0.00, 0.10, 0.60]
MULT = {
    "null":   {"CBS": 0.4, "MOBILE_APP": 0.8, "BRANCH_PORTAL": 2.5, "LOAN_ORIGINATION": 1.0},
    "mobile": {"CBS": 0.5, "MOBILE_APP": 2.5, "BRANCH_PORTAL": 1.5, "LOAN_ORIGINATION": 0.8},
    "text":   {"CBS": 0.4, "MOBILE_APP": 0.8, "BRANCH_PORTAL": 2.8, "LOAN_ORIGINATION": 0.9},
    "pan":    {"CBS": 0.5, "MOBILE_APP": 0.6, "BRANCH_PORTAL": 1.0, "LOAN_ORIGINATION": 3.0},
}


# ------------------------------------------------------------------ helpers
def rand_dates(n, lo, hi):
    return lo + pd.to_timedelta(rng.integers(0, (hi - lo).days + 1, n), unit="D")


def z(a, w):
    return np.char.zfill(np.asarray(a).astype(str), w)


def objcols(df):
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(object)
    return df


def to_str(s):
    s = pd.Series(s).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d").fillna("").values
    return s.astype(object).where(s.notna(), "").astype(str).values


def unique_ids(n, lo, hi):
    ids = np.unique(rng.integers(lo, hi, int(n * 1.1) + 100))
    rng.shuffle(ids)
    return ids[:n]


LOG, TOUCHED = [], {}


def log_rows(table, ids, col, etype, dim, old, new, src, batch, notes=""):
    LOG.append(pd.DataFrame({
        "table_name": table, "record_id": np.asarray(ids), "column_name": col,
        "error_type": etype, "dq_dimension": dim,
        "original_value": to_str(old), "injected_value": to_str(new),
        "source_system": np.asarray(src), "batch_id": np.asarray(batch), "notes": notes}))


def inject(df, table, idcol, col, p, etype, dim, newfn, fam=None, mask=None, notes=""):
    n = len(df)
    if mask is None:
        pr = np.full(n, float(p))
        if fam and "source_system" in df:
            pr = p * df["source_system"].map(MULT[fam]).astype(float).values
        mask = rng.random(n) < pr
    touched = TOUCHED.setdefault((table, col), np.zeros(n, bool))
    mask = np.asarray(mask) & ~touched
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    touched[idx] = True
    old = df[col].iloc[idx].copy()
    new = pd.Series(newfn(old, idx))
    df.iloc[idx, df.columns.get_loc(col)] = new.values
    src = df["source_system"].iloc[idx].values if "source_system" in df else [""] * len(idx)
    bat = df["batch_id"].iloc[idx].values if "batch_id" in df else [""] * len(idx)
    log_rows(table, df[idcol].iloc[idx].values, col, etype, dim, old, new, src, bat, notes)


def add_source(df, src_p, event_col=None):
    n = len(df)
    df["source_system"] = rng.choice(SOURCES, n, p=src_p).astype(object)
    ld = pd.Series(rand_dates(n, START, END))
    if event_col is not None:                      # new records are loaded the next day
        ev = df[event_col]
        recent = (ev >= START) & (ev < END)
        ld = ld.where(~recent, ev + DAY)
    df["load_date"] = ld
    df["batch_id"] = (df["source_system"] + "-" + df["load_date"].dt.strftime("%Y%m%d")).astype(object)


# ------------------------------------------------------------------ reference data: localities + PINs
# (city: state, population weight, "Area:PIN;...")  -- PINs are for realism; verify against
# the India Post PIN directory before using them for anything other than test data.
CITY_DATA = {
    "Bengaluru": ("Karnataka", 10, "Koramangala:560034;Indiranagar:560038;Jayanagar:560011;Whitefield:560066;HSR Layout:560102;Malleshwaram:560003;Basavanagudi:560004;Electronic City:560100;JP Nagar:560078;Yelahanka:560064;Rajajinagar:560010;BTM Layout:560076;Marathahalli:560037;Hebbal:560024;Banashankari:560070;Vijayanagar:560040"),
    "Mumbai": ("Maharashtra", 12, "Andheri West:400058;Bandra West:400050;Borivali West:400092;Dadar West:400028;Powai:400076;Colaba:400005;Malad West:400064;Goregaon East:400063;Ghatkopar East:400077;Chembur:400071;Vile Parle West:400056;Kandivali West:400067;Mulund West:400080;Worli:400018"),
    "Delhi": ("Delhi", 12, "Connaught Place:110001;Karol Bagh:110005;Lajpat Nagar:110024;Dwarka:110075;Rohini:110085;Saket:110017;Janakpuri:110058;Pitampura:110034;Vasant Kunj:110070;Laxmi Nagar:110092;Greater Kailash:110048;Mayur Vihar:110091;Nehru Place:110019;Paschim Vihar:110063"),
    "Chennai": ("Tamil Nadu", 8, "T Nagar:600017;Adyar:600020;Anna Nagar:600040;Velachery:600042;Tambaram:600045;Mylapore:600004;Porur:600116;Perambur:600011;Guindy:600032;Nungambakkam:600034"),
    "Hyderabad": ("Telangana", 8, "Banjara Hills:500034;Jubilee Hills:500033;Madhapur:500081;Gachibowli:500032;Secunderabad:500003;Kukatpally:500072;Ameerpet:500016;Dilsukhnagar:500060;Kondapur:500084;Begumpet:500016"),
    "Kolkata": ("West Bengal", 7, "Salt Lake:700091;Park Street:700016;Ballygunge:700019;Behala:700034;Dum Dum:700028;Jadavpur:700032;New Town:700156;Garia:700084;Tollygunge:700033"),
    "Pune": ("Maharashtra", 6, "Kothrud:411038;Viman Nagar:411014;Hinjewadi:411057;Shivajinagar:411005;Baner:411045;Hadapsar:411028;Aundh:411007;Kharadi:411014;Pimpri:411018"),
    "Ahmedabad": ("Gujarat", 5, "Navrangpura:380009;Satellite:380015;Maninagar:380008;Bopal:380058;Vastrapur:380015;Naranpura:380013;Paldi:380007"),
    "Jaipur": ("Rajasthan", 4, "Malviya Nagar:302017;Vaishali Nagar:302021;C Scheme:302001;Mansarovar:302020;Tonk Road:302015;Jagatpura:302017"),
    "Lucknow": ("Uttar Pradesh", 4, "Gomti Nagar:226010;Hazratganj:226001;Aliganj:226024;Indira Nagar:226016;Alambagh:226005"),
    "Kochi": ("Kerala", 3, "Ernakulam:682011;Edappally:682024;Kakkanad:682030;Palarivattom:682025;Vyttila:682019"),
    "Chandigarh": ("Chandigarh", 2, "Sector 17:160017;Sector 22:160022;Sector 35:160035;Sector 43:160043"),
    "Bhopal": ("Madhya Pradesh", 2, "Arera Colony:462016;MP Nagar:462011;Kolar Road:462042"),
    "Patna": ("Bihar", 3, "Boring Road:800001;Kankarbagh:800020;Rajendra Nagar:800016"),
    "Indore": ("Madhya Pradesh", 3, "Vijay Nagar:452010;Palasia:452001;Rajwada:452002"),
    "Nagpur": ("Maharashtra", 2, "Dharampeth:440010;Sitabuldi:440012"),
    "Surat": ("Gujarat", 3, "Adajan:395009;Vesu:395007;Varachha:395006"),
    "Coimbatore": ("Tamil Nadu", 2, "RS Puram:641002;Peelamedu:641004;Gandhipuram:641012"),
    "Visakhapatnam": ("Andhra Pradesh", 2, "MVP Colony:530017;Dwaraka Nagar:530016"),
    "Guwahati": ("Assam", 2, "Dispur:781006;Beltola:781028;Paltan Bazar:781008"),
}
REGION = {"Karnataka": "SOUTH", "Tamil Nadu": "SOUTH", "Kerala": "SOUTH", "Telangana": "SOUTH",
          "Andhra Pradesh": "SOUTH", "Maharashtra": "WEST", "Gujarat": "WEST", "Delhi": "NORTH",
          "Uttar Pradesh": "NORTH", "Rajasthan": "NORTH", "Chandigarh": "NORTH", "Bihar": "NORTH",
          "Madhya Pradesh": "NORTH", "West Bengal": "EAST", "Assam": "EAST"}

rows = []
for city, (state, _w, s) in CITY_DATA.items():
    for item in s.split(";"):
        area, pin = item.rsplit(":", 1)
        rows.append((pin, area, city, state))
pin_ref = pd.DataFrame(rows, columns=["pin_code", "area_name", "city", "state"])
NB = len(pin_ref)
city_w = {c: v[1] for c, v in CITY_DATA.items()}
pw = pin_ref.city.map(city_w).astype(float).values / pin_ref.groupby("city")["city"].transform("size").values
pw = pw / pw.sum()
pos_in_city = pin_ref.groupby("city", sort=False).cumcount().values
city_start = np.arange(NB) - pos_in_city
city_len = pin_ref.groupby("city", sort=False)["city"].transform("size").values

# ------------------------------------------------------------------ reference data: names
NAMES = {
    "SOUTH": {
        "M": "Karthik Suresh Ramesh Venkatesh Srinivas Arun Balaji Harish Naveen Prakash Vijay Ganesh Manoj Sathish Anand Ravi Mohan Kiran Sandeep Raghavan Vignesh Aravind Deepak Santhosh Praveen Mahesh Lokesh Rajesh Krishna Ashwin Sudhir Girish Vinod Shankar Gopal Jayaram Madhavan Hariharan Bharath".split(),
        "F": "Lakshmi Priya Divya Anitha Kavitha Sowmya Deepa Meena Padma Revathi Swathi Sangeetha Nandini Bhavani Shobha Vijaya Radhika Geetha Uma Jayanthi Sunitha Pavithra Aishwarya Harini Keerthi Ramya Sneha Lavanya Manjula Vasanthi Shruthi Archana Malathi Rajeshwari Bindu Latha Anjali Nithya Gayathri".split(),
        "L": "Reddy Naidu Iyer Iyengar Nair Menon Pillai Gowda Shetty Rao Murthy Krishnan Subramanian Raman Venkatesh Prasad Kurup Varma Hegde Bhat Kamath Chandran Narayanan Sundaram Balasubramanian Ramachandran Srinivasan Natarajan Rajan Nambiar".split()},
    "WEST": {
        "M": "Amit Rahul Sachin Nitin Prashant Sandeep Swapnil Omkar Rohan Vishal Mangesh Ashish Hardik Jignesh Kalpesh Paresh Chirag Dhaval Nilesh Mihir Yash Kunal Tushar Vikram Sunil Anil Ajay Pratik Siddharth Kedar Viraj Bhavesh Jayesh Mitesh Rakesh Harshad Ketan Bhargav Parth Jitendra".split(),
        "F": "Sneha Pooja Neha Priyanka Snehal Rutuja Shweta Madhuri Sayali Aarti Swati Kavita Jyoti Manisha Komal Nidhi Hetal Krupa Dipika Bhavna Rupal Payal Ritika Shilpa Vaishali Sonal Mansi Jinal Foram Khushbu Pallavi Supriya Anagha Gauri Mrunal Trupti Ketaki Disha Nikita Purvi".split(),
        "L": "Patil Deshmukh Kulkarni Jadhav Pawar Shinde More Gaikwad Joshi Kadam Chavan Bhosale Sawant Deshpande Naik Patel Shah Mehta Desai Trivedi Parekh Modi Doshi Gandhi Thakkar Vora Amin Chokshi Bhatt Dave Pandya Solanki Rathod Zaveri Sheth Kothari Kamdar Dalal Vyas".split()},
    "NORTH": {
        "M": "Rajesh Amit Vikas Sunil Anil Manoj Sanjay Deepak Rohit Ankit Rahul Nitin Pankaj Ashok Vijay Mukesh Naveen Gaurav Sumit Mohit Ajay Arvind Dinesh Pradeep Lalit Yogesh Vivek Rakesh Sandeep Kapil Varun Tarun Abhishek Shivam Harsh Akash Devendra Umesh Surendra Brijesh".split(),
        "F": "Sunita Anita Rekha Meena Kavita Pooja Neha Priya Anjali Nisha Ritu Seema Geeta Sapna Shweta Swati Deepika Preeti Renu Sarita Manju Poonam Savita Kiran Archana Vandana Shalini Divya Aarti Ruchi Pallavi Richa Tanvi Komal Megha Nikita Shikha Jyoti Payal Ishita".split(),
        "L": "Sharma Verma Gupta Singh Yadav Mishra Pandey Tiwari Dubey Chauhan Agarwal Jain Kapoor Malhotra Mehra Khanna Arora Chopra Bansal Goyal Saxena Srivastava Rastogi Tandon Bhardwaj Rawat Negi Joshi Thakur Kumar Prasad Jha Sinha Choudhary Shukla Garg Mittal Kohli Bhatia Sethi".split()},
    "EAST": {
        "M": "Subhash Sourav Arnab Debashish Abhijit Sudipto Anirban Partha Soumya Rajib Tapan Biswajit Sandip Amit Sujit Prasenjit Ranjit Kaushik Somnath Pranab Dipankar Arijit Ashim Bhaskar Manas Pritam Ritwik Sayan Rupam Jayanta Bikash Hemanta Kamal Ratan Utpal Gautam Tanmoy".split(),
        "F": "Sumita Anindita Moumita Debjani Rima Sharmila Papiya Madhumita Tanushree Sudeshna Piyali Mitali Rupa Sanchita Payel Nabanita Swati Arpita Shreya Ananya Susmita Dolon Mahua Ipsita Rituparna Baisakhi Sagarika Soma Barnali Chandana Ruma Jhuma Aparna Nandita Bidisha Trisha Lipika Sampa Kakoli".split(),
        "L": "Banerjee Chatterjee Mukherjee Bose Sen Das Dutta Ghosh Roy Chowdhury Bhattacharya Sarkar Basu Mondal Saha Paul Dey Majumdar Ganguly Biswas Sengupta Halder Pal Guha Mitra Nandi Kar Bora Baruah Gogoi Hazarika Saikia".split()},
    "MUSLIM": {
        "M": "Mohammed Imran Faisal Irfan Salman Arif Javed Rashid Shahid Aamir Zaheer Farhan Asif Nadeem Tariq Yusuf Adnan Sameer Junaid Rizwan".split(),
        "F": "Ayesha Fatima Farah Shabana Nazia Saima Rukhsar Zoya Sana Nasreen Shabnam Afreen Shaista Rubina Tabassum Mehnaz Sadiya Uzma Zainab Hina".split(),
        "L": "Khan Ansari Qureshi Sheikh Siddiqui Hussain Syed Pathan Mirza Malik Rizvi Farooqui Naqvi Akhtar Ali".split()},
    "CHRISTIAN": {
        "M": "John Joseph Thomas George Joshua Samuel Daniel Mathew Jacob Antony Francis Xavier Peter Paul Sebastian".split(),
        "F": "Mary Susan Anita Grace Sheela Jessy Ruth Elizabeth Sarah Rachel Jennifer Lissy Mercy Teresa Annie".split(),
        "L": "Thomas Joseph George Mathew Varghese Fernandes Pereira Rodrigues Dias Lobo Mascarenhas Abraham Philip Kurian".split()},
    "SIKH": {
        "M": "Harpreet Gurpreet Manpreet Jaspreet Amarjeet Balwinder Gurmeet Sukhdev Jasbir Harjinder Kuldeep Rajinder Navdeep Parminder Gagandeep".split(),
        "F": "Harpreet Manpreet Simran Jasleen Gurleen Navneet Amandeep Kiranjeet Rajwinder Paramjeet Sukhmani Harleen".split(),
        "L": "Singh Gill Sandhu Dhillon Sidhu Grewal Bajwa Brar Bedi Ahluwalia Walia Chahal Randhawa".split()},
}
ALL_FIRST = sorted({x for v in NAMES.values() for k in ("M", "F") for x in v[k]})
ALL_LAST = sorted({x for v in NAMES.values() for x in v["L"]})


def make_names(states, gender):
    n = len(states)
    region = pd.Series(states).map(REGION).values.astype(object)
    u = rng.random(n)
    p_m = 0.09
    p_c = np.where(region == "SOUTH", 0.10, 0.02)
    p_s = pd.Series(states).map({"Chandigarh": 0.35, "Delhi": 0.07}).fillna(0.01).values
    pool = region.copy()
    pool[u < p_m] = "MUSLIM"
    pool[(u >= p_m) & (u < p_m + p_c)] = "CHRISTIAN"
    pool[(u >= p_m + p_c) & (u < p_m + p_c + p_s)] = "SIKH"
    first, last = np.empty(n, object), np.empty(n, object)
    for pl in set(pool):
        for g in ("M", "F"):
            m = (pool == pl) & (gender == g)
            k = int(m.sum())
            if k:
                first[m] = rng.choice(NAMES[pl][g], k)
                last[m] = rng.choice(NAMES[pl]["L"], k)
    last[(pool == "SIKH") & (gender == "F") & (rng.random(n) < 0.75)] = "Kaur"
    last[(pool == "SIKH") & (gender == "M") & (rng.random(n) < 0.5)] = "Singh"
    return first, last


def name_variant(full):
    f, l = full.split(" ", 1)
    r = rng.random()
    if r < 0.35:
        return f"{f[0]} {l}"
    if r < 0.60:
        i = int(rng.integers(1, len(l)))
        return f"{f} {l[:i]}{l[i]}{l[i:]}"
    if r < 0.80:
        return f"{l} {f}"
    if r < 0.90:
        return f"{f.upper()} {l.upper()}"
    return f"{f[0]}{f[2]}{f[1]}{f[3:]} {l}" if len(f) > 3 else f"{f[0]} {l}"


# ------------------------------------------------------------------ branches
codes = np.sort(rng.choice(np.arange(1001, 9999), NB, replace=False))
branch = pd.DataFrame({
    "branch_code": [f"{c:04d}" for c in codes],
    "branch_name": (pin_ref.area_name + " Branch").values,
    "ifsc": [f"{BANK_CODE}0{c:06d}" for c in codes],
    "micr_code": [f"{p[:3]}{BANK_MICR}{c % 1000:03d}" for p, c in zip(pin_ref.pin_code, codes)],
    "address": (pin_ref.area_name + " - " + pin_ref.city).values,
    "city": pin_ref.city.values, "state": pin_ref.state.values, "pin_code": pin_ref.pin_code.values,
    "opened_date": rand_dates(NB, pd.Timestamp("1990-01-01"), pd.Timestamp("2022-12-31")),
})
objcols(branch)
BR_CODES = branch.branch_code.values
BR_IFSC = dict(zip(branch.branch_code, branch.ifsc))

# ------------------------------------------------------------------ customers
dup_rate = DUP_RATE if INJECT_ERRORS else 0.0
n0 = int(round(N_CUSTOMERS / (1 + dup_rate)))
nd = N_CUSTOMERS - n0
ids_all = unique_ids(N_CUSTOMERS, 3_000_000_000, 8_999_999_999)     # 10-digit CIF numbers

pi = rng.choice(NB, n0, p=pw)
states = pin_ref.state.values[pi]
gender = rng.choice(np.array(["M", "F"], dtype=object), n0, p=[0.52, 0.48])
first, last = make_names(states, gender)
full = np.array([f"{a} {b}" for a, b in zip(first, last)], dtype=object)

age = np.clip(rng.normal(39, 13, n0), 18.0, 84.0)
dob = pd.Series(END - pd.to_timedelta(np.round(age * 365.25 + rng.uniform(0, 365, n0)), unit="D"))
age = ((END - dob).dt.days / 365.25).values

occ = np.empty(n0, object)
young, old = age < 23, age >= 60
mid = ~young & ~old
occ[young] = rng.choice(["STUDENT", "SALARIED", "SELF_EMPLOYED"], young.sum(), p=[.5, .4, .1])
occ[old] = rng.choice(["RETIRED", "BUSINESS", "SELF_EMPLOYED", "HOMEMAKER", "PROFESSIONAL"], old.sum(),
                      p=[.55, .15, .10, .12, .08])
occ[mid] = rng.choice(["SALARIED", "SELF_EMPLOYED", "BUSINESS", "PROFESSIONAL", "HOMEMAKER", "AGRICULTURIST"],
                      mid.sum(), p=[.52, .14, .12, .08, .10, .04])
occ[(occ == "HOMEMAKER") & (gender == "M")] = "SALARIED"
INC = {"SALARIED": (13.1, .6), "SELF_EMPLOYED": (13.2, .7), "BUSINESS": (13.6, .8),
       "PROFESSIONAL": (14.0, .6), "RETIRED": (12.7, .5), "AGRICULTURIST": (12.4, .5)}
income = np.zeros(n0)
for k, (m, s) in INC.items():
    mk = occ == k
    income[mk] = np.maximum(60000, np.round(rng.lognormal(m, s, mk.sum()), -3))

marital = np.where(age < 24, rng.choice(["SINGLE", "MARRIED"], n0, p=[.93, .07]),
                   rng.choice(["MARRIED", "SINGLE", "WIDOWED", "DIVORCED"], n0, p=[.80, .12, .05, .03]))

PREFIX2 = np.array([98, 99, 97, 96, 95, 94, 93, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77,
                    76, 75, 74, 73, 72, 70, 63, 62, 60])


def gen_mobiles(n):
    return rng.choice(PREFIX2, n).astype(np.int64) * 10 ** 8 + rng.integers(0, 10 ** 8, n)


def unique_mobiles(n):
    m = gen_mobiles(n)
    for _ in range(10):
        dup = pd.Series(m).duplicated(keep="first").values
        if not dup.any():
            break
        m[dup] = gen_mobiles(int(dup.sum()))
    return m.astype(str).astype(object)


mobile = unique_mobiles(n0)

f_l, l_l = np.char.lower(first.astype(str)), np.char.lower(last.astype(str))
yy = z(dob.dt.year.values % 100, 2)
dg = rng.integers(1, 99, n0).astype(str)
pat = rng.integers(0, 7, n0)
loc = np.select(
    [pat == 0, pat == 1, pat == 2, pat == 3, pat == 4, pat == 5],
    [np.char.add(np.char.add(f_l, "."), l_l), np.char.add(f_l, l_l), np.char.add(np.char.add(f_l, "_"), l_l),
     np.char.add(np.char.add(f_l.astype("U1"), "."), l_l), np.char.add(np.char.add(np.char.add(f_l, "."), l_l), yy),
     np.char.add(f_l, yy)],
    default=np.char.add(np.char.add(f_l, l_l), dg))
loc = loc.astype(object)
dom = rng.choice(["gmail.com", "yahoo.co.in", "outlook.com", "rediffmail.com", "hotmail.com", "icloud.com"], n0,
                 p=[.68, .10, .08, .05, .05, .04])
email = pd.Series(loc + "@" + dom)
for _ in range(10):
    dp = email.duplicated(keep="first").values
    if not dp.any():
        break
    loc[dp] = loc[dp] + rng.integers(1, 999, int(dp.sum())).astype(str)
    email = pd.Series(loc + "@" + dom)
email = email.values

LET = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def gen_pan(lastnames):
    n = len(lastnames)
    a = np.char.add(np.char.add(LET[rng.integers(0, 26, n)], LET[rng.integers(0, 26, n)]), LET[rng.integers(0, 26, n)])
    b = np.char.add(np.char.add(a, "P"), np.char.upper(np.asarray(lastnames).astype(str).astype("U1")))
    return np.char.add(np.char.add(b, z(rng.integers(0, 10000, n), 4)), LET[rng.integers(0, 26, n)]).astype(object)


pan = gen_pan(last)
for _ in range(10):
    dp = pd.Series(pan).duplicated(keep="first").values
    if not dp.any():
        break
    pan[dp] = gen_pan(last[dp])

BLD = ["Sai Krupa", "Shanti Residency", "Lakshmi Nivas", "Green Park Apartments", "Sunrise Heights", "Rajshree Towers",
       "Ganga Enclave", "Krishna Kunj", "Vasant Villa", "Ashirwad Apartments", "Om Sai Residency", "Maple Court",
       "Royal Palms", "Jai Hind Apartments", "Sagar Darshan", "Nandanvan Society", "Shree Ganesh Complex"]
STR = ["MG Road", "Gandhi Road", "Nehru Marg", "Temple Street", "Station Road", "Market Road", "Lake View Road",
       "Church Road", "Ring Road", "Main Road", "College Road", "Residency Road", "Hospital Road", "Canal Road"]
CRS = ["1st Cross", "2nd Cross", "3rd Cross", "4th Main", "5th Main", "6th Cross", "7th Main", "8th Cross", "10th Main"]
tpl = rng.integers(0, 6, n0)
h, h2 = rng.integers(1, 250, n0), rng.integers(1, 60, n0)
fl = rng.integers(1, 9, n0) * 100 + rng.integers(1, 9, n0)
b_, s_, c_ = rng.choice(BLD, n0), rng.choice(STR, n0), rng.choice(CRS, n0)
addr1 = np.array([
    [f"Flat {fl[i]} {b_[i]}", f"H No {h[i]} {s_[i]}", f"{h[i]}/{h2[i]} {s_[i]}", f"No {h[i]} {c_[i]} {s_[i]}",
     f"Door No {h[i]} {s_[i]}", f"Plot {h[i]} {b_[i]}"][tpl[i]] for i in range(n0)], dtype=object)

dob18 = dob + pd.Timedelta(days=6575)
low = dob18.where(dob18 > pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-01"))
span = ((END - DAY) - low).dt.days.clip(lower=0)
since = low + pd.to_timedelta(np.floor(rng.random(n0) * (span.values + 1)), unit="D")
rec_d = pd.Series(rand_dates(n0, START, END - DAY))
since = since.where(~((rng.random(n0) < 0.12) & (rec_d >= low)), rec_d).clip(upper=END - DAY)
kyc_date = (since + pd.to_timedelta(np.floor(rng.random(n0) * ((END - since).dt.days.values + 1)), unit="D")).clip(upper=END)

own = rng.random(n0) < 0.55
home_idx = np.where(own, pi, city_start[pi] + rng.integers(0, city_len[pi]))

cust = pd.DataFrame({
    "customer_id": ids_all[:n0].astype(str).astype(object),
    "full_name": full, "dob": dob, "gender": gender, "marital_status": marital.astype(object),
    "mobile": mobile, "email": email, "address_line1": addr1,
    "address_line2": pin_ref.area_name.values[pi], "city": pin_ref.city.values[pi], "state": states,
    "pin_code": pin_ref.pin_code.values[pi], "pan": pan,
    "aadhaar_last4": z(rng.integers(0, 10000, n0), 4).astype(object),
    "occupation": occ, "annual_income": income.astype(np.int64),
    "kyc_status": rng.choice(["VERIFIED", "PENDING", "EXPIRED"], n0, p=[.93, .05, .02]).astype(object),
    "kyc_date": kyc_date, "customer_since": since, "home_branch_code": BR_CODES[home_idx],
})
objcols(cust)

if nd:                                       # duplicate customers (same person, re-registered)
    src_idx = rng.choice(n0, nd, replace=False)
    d = cust.iloc[src_idx].copy().reset_index(drop=True)
    d["customer_id"] = ids_all[n0:].astype(str).astype(object)
    orig_names = d["full_name"].copy()
    d["full_name"] = d["full_name"].map(name_variant)
    dlow = (pd.to_datetime(d["dob"]) + pd.Timedelta(days=6575)).clip(lower=START)
    d["customer_since"] = (dlow + pd.to_timedelta(np.floor(rng.random(nd) * ((END - DAY - dlow).dt.days.clip(lower=0).values + 1)), unit="D")).clip(upper=END - DAY)
    d["kyc_date"] = d["customer_since"]
    dup_of = cust["customer_id"].iloc[src_idx].values
    cust = pd.concat([cust, d], ignore_index=True)
    add_source(cust, SRC_P_CUST, "customer_since")
    is_dup = np.zeros(len(cust), bool)
    is_dup[n0:] = True
    TOUCHED[("customer_master", "full_name")] = is_dup.copy()
    log_rows("customer_master", d["customer_id"].values, "full_name", "DUPLICATE_CUSTOMER", "UNIQUENESS",
             orig_names, d["full_name"], cust["source_system"].iloc[n0:].values, cust["batch_id"].iloc[n0:].values,
             [f"duplicate_of={x}" for x in dup_of])
else:
    add_source(cust, SRC_P_CUST, "customer_since")
n_c = len(cust)
cust_dob = pd.to_datetime(cust["dob"]).reset_index(drop=True)
cust_age = ((END - cust_dob).dt.days / 365.25).values

# ------------------------------------------------------------------ accounts
n_extra = int(n_c * EXTRA_ACCT_RATIO)
NA = n_c + n_extra
occ_c = cust["occupation"].values
prim = np.empty(n_c, object)
sal = occ_c == "SALARIED"
biz = np.isin(occ_c, ["BUSINESS", "SELF_EMPLOYED", "PROFESSIONAL"])
prim[sal] = rng.choice(["SALARY", "SAVINGS"], sal.sum(), p=[.55, .45])
prim[biz] = rng.choice(["CURRENT", "SAVINGS"], biz.sum(), p=[.35, .65])
prim[~sal & ~biz] = "SAVINGS"
cidx = np.concatenate([np.arange(n_c), rng.integers(0, n_c, n_extra)])
atype = np.concatenate([prim, rng.choice(["FIXED_DEPOSIT", "SAVINGS"], n_extra, p=[.5, .5])]).astype(object)
cs = pd.Series(pd.to_datetime(cust["customer_since"]).values[cidx])
is_prim = np.arange(NA) < n_c
open_dt = pd.Series(np.where(is_prim, (cs + pd.to_timedelta(rng.integers(0, 3, NA), unit="D")).values,
                             (cs + pd.to_timedelta(np.floor(rng.random(NA) * ((END - cs).dt.days.values + 1)),
                                                   unit="D")).values))
open_dt = open_dt.clip(upper=END - DAY)
a_branch = cust["home_branch_code"].values[cidx]
PROD = {"SAVINGS": "01", "SALARY": "02", "CURRENT": "11", "FIXED_DEPOSIT": "31"}
a_prod = np.array([PROD[t] for t in atype], dtype=object)
order = np.argsort(open_dt.values, kind="stable")
cc = pd.DataFrame({"b": a_branch[order], "p": a_prod[order]}).groupby(["b", "p"]).cumcount().values
seq = np.empty(NA, np.int64)
seq[order] = 100000 + cc
acct_no = np.array([f"{b}{p}{s:06d}" for b, p, s in zip(a_branch, a_prod, seq)], dtype=object)   # 12 digits

inc_m = cust["annual_income"].values[cidx] / 12.0
bal = np.zeros(NA)
for t, (fl_, m, s) in {"SAVINGS": (15000, .3, .9), "SALARY": (15000, .3, .9), "CURRENT": (50000, .8, 1.0)}.items():
    mk = atype == t
    bal[mk] = np.maximum(inc_m[mk], fl_) * rng.lognormal(m, s, mk.sum())
mk = atype == "FIXED_DEPOSIT"
bal[mk] = np.maximum(10000, np.round(rng.lognormal(12.0, .9, mk.sum()), -3))
bal = np.round(bal, 2)
status = np.where(atype == "FIXED_DEPOSIT", rng.choice(["ACTIVE", "CLOSED"], NA, p=[.92, .08]),
                  rng.choice(["ACTIVE", "DORMANT", "CLOSED"], NA, p=[.91, .06, .03])).astype(object)
status[atype == "SALARY"] = np.where(rng.random((atype == "SALARY").sum()) < 0.97, "ACTIVE", "CLOSED")

acct = pd.DataFrame({
    "account_number": acct_no,
    "customer_id": cust["customer_id"].values[cidx],
    "account_type": atype, "branch_code": a_branch,
    "ifsc": [BR_IFSC[b] for b in a_branch],
    "open_date": open_dt, "balance": bal, "currency": "INR", "status": status})
objcols(acct)
add_source(acct, SRC_P_CUST, "open_date")
primary_acct = acct_no[:n_c]

# ------------------------------------------------------------------ loans
LT = ["HOME", "PERSONAL", "VEHICLE", "EDUCATION", "BUSINESS", "GOLD"]
LP = {"HOME": (15.2, .5, [120, 180, 240, 300], 8.35, 9.6, "61", -4),
      "PERSONAL": (12.4, .6, [12, 24, 36, 48, 60], 11.0, 16.0, "62", -3),
      "VEHICLE": (13.2, .4, [36, 48, 60, 84], 8.7, 11.0, "63", -4),
      "EDUCATION": (13.1, .5, [60, 84, 120], 8.5, 11.0, "64", -4),
      "BUSINESS": (14.2, .8, [36, 60, 84, 120], 10.0, 14.0, "65", -4),
      "GOLD": (11.9, .5, [6, 12, 24], 8.5, 10.0, "66", -3)}
NL = N_LOANS
elig = np.isin(cust["occupation"].values, ["SALARIED", "SELF_EMPLOYED", "BUSINESS", "PROFESSIONAL"]) & (cust_age >= 22) & (cust_age <= 65)
w = np.where(elig, 1.0, 0.15)
lidx = rng.choice(n_c, NL, p=w / w.sum())
ltype = rng.choice(LT, NL, p=[.20, .30, .20, .10, .10, .10]).astype(object)
ltype[(ltype == "EDUCATION") & (cust_age[lidx] >= 45)] = "PERSONAL"
lamt, lten, lrate, lprod = np.zeros(NL), np.zeros(NL, np.int64), np.zeros(NL), np.empty(NL, object)
for t, (m, s, tens, r0, r1, pc, rd) in LP.items():
    mk = ltype == t
    k = int(mk.sum())
    lamt[mk] = np.maximum(20000, np.round(rng.lognormal(m, s, k), rd))
    lten[mk] = rng.choice(tens, k)
    lrate[mk] = np.round(rng.uniform(r0, r1, k) * 20) / 20
    lprod[mk] = pc
lo_d = pd.concat([pd.Series(pd.to_datetime(cust["customer_since"]).values[lidx]),
                  cust_dob.iloc[lidx].reset_index(drop=True) + pd.Timedelta(days=7670),
                  pd.Series([pd.Timestamp("2015-01-01")] * NL)], axis=1).max(axis=1)
span_d = ((END - DAY) - lo_d).dt.days.clip(lower=0)
disb = lo_d + pd.to_timedelta(np.floor(rng.random(NL) * (span_d.values + 1)), unit="D")
rd_ = pd.Series(rand_dates(NL, START, END - DAY))
disb = disb.where(rng.random(NL) >= 0.35, pd.concat([rd_, lo_d], axis=1).max(axis=1)).clip(upper=END - DAY)
emi_day = rng.integers(1, 29, NL)
dm = disb.dt.year.values * 12 + disb.dt.month.values - 1
mm = dm + lten
mat = pd.to_datetime(pd.DataFrame({"year": mm // 12, "month": mm % 12 + 1, "day": emi_day}))
r = lrate / 1200.0
emi = np.round(lamt * r * (1 + r) ** lten / ((1 + r) ** lten - 1), 0)
em = END.year * 12 + END.month - 1
paid = np.clip(em - dm - (END.day < emi_day).astype(int), 0, lten)
out_p = np.where(paid >= lten, 0.0, np.round(lamt * ((1 + r) ** lten - (1 + r) ** paid) / ((1 + r) ** lten - 1), 2))
lstat = np.where(paid >= lten, "CLOSED", "ACTIVE").astype(object)
dpd = np.zeros(NL, np.int64)
act = lstat == "ACTIVE"
u = rng.random(NL)
sma = act & (u < 0.04)
npa = act & (u >= 0.04) & (u < 0.065)
dpd[sma] = rng.integers(1, 90, sma.sum())
dpd[npa] = rng.integers(91, 400, npa.sum())
lstat[npa] = "NPA"
l_branch = cust["home_branch_code"].values[lidx]
lord = np.argsort(disb.values, kind="stable")
lcc = pd.DataFrame({"b": l_branch[lord], "p": lprod[lord]}).groupby(["b", "p"]).cumcount().values
lseq = np.empty(NL, np.int64)
lseq[lord] = 10000001 + lcc
loan = pd.DataFrame({
    "loan_account_no": [f"{b}{p}{s:08d}" for b, p, s in zip(l_branch, lprod, lseq)],       # 14 digits
    "customer_id": cust["customer_id"].values[lidx], "loan_type": ltype, "branch_code": l_branch,
    "disbursement_date": disb, "disbursement_amount": lamt, "tenure_months": lten, "interest_rate": lrate,
    "emi_amount": emi, "emi_account_number": primary_acct[lidx], "maturity_date": mat,
    "outstanding_principal": out_p, "overdue_days": dpd, "status": lstat, "_emi_day": emi_day})
objcols(loan)
add_source(loan, SRC_P_LOAN, "disbursement_date")

# loan customers must have a live account to pay EMIs from
force = acct.account_number.isin(set(loan.emi_account_number))
acct.loc[force & (acct.status != "ACTIVE"), "status"] = "ACTIVE"
acct.loc[force & (acct.balance <= 0), "balance"] = np.round(rng.uniform(5000, 60000, int((force & (acct.balance <= 0)).sum())), 2)

# ------------------------------------------------------------------ transactions
UPI_M = ["SWIGGY", "ZOMATO", "AMAZON PAY", "FLIPKART", "BIGBASKET", "BLINKIT", "UBER INDIA", "OLA CABS", "IRCTC",
         "BOOKMYSHOW", "AIRTEL PAYMENTS", "JIO PREPAID", "DMART", "MEDPLUS PHARMACY", "APOLLO PHARMACY", "INDIAN OIL",
         "CHAI POINT", "DOMINOS PIZZA", "MYNTRA", "NETFLIX", "SPOTIFY", "BESCOM"]
POS_M = ["RELIANCE RETAIL", "DMART", "MORE SUPERMARKET", "LIFESTYLE STORES", "WESTSIDE", "SHOPPERS STOP", "CROMA",
         "VIJAY SALES", "APOLLO PHARMACY", "HP PETROL PUMP", "INDIAN OIL FUEL", "CAFE COFFEE DAY", "PIZZA HUT",
         "DECATHLON", "PANTALOONS", "TANISHQ JEWELLERS"]
BILLERS = ["BESCOM", "TATA POWER", "MSEB", "BSES DELHI", "AIRTEL POSTPAID", "JIO FIBER", "LIC PREMIUM", "ACT FIBERNET",
           "BSNL BROADBAND", "INDANE GAS", "MAHANAGAR GAS", "HATHWAY BROADBAND"]
EMPLOYERS = ["NEXORA TECHNOLOGIES PVT LTD", "BLUEPEAK SOFTWARE SERVICES", "ORBITAL SYSTEMS PVT LTD", "SUMMIT INFRA LTD",
             "CRESTVIEW CONSULTING", "AURORA HEALTHCARE", "PIONEER LOGISTICS PVT LTD", "KESARI FOODS LTD",
             "LOTUS MANUFACTURING", "MERIDIAN SERVICES PVT LTD", "HORIZON EDUTECH", "VERTEX ENGINEERING WORKS",
             "TRIDENT RETAIL PVT LTD", "SILVERLINE TELECOM", "GREENFIELD AGRO LTD"]
HANDLES = ["ybl", "okhdfcbank", "paytm", "oksbi", "okaxis", "ibl", "axl", "okicici"]
CITIES_UP = [c.upper() for c in CITY_DATA]
BANKS4 = ["SBIN", "HDFC", "ICIC", "UTIB", "KKBK", "PUNB", "BARB", "CNRB"]
CH = ["UPI", "ATM", "POS", "NET_BANKING", "IMPS", "NEFT", "BRANCH"]
CH_P = [.42, .08, .17, .08, .06, .10, .09]
P_DEBIT = {"UPI": .72, "ATM": 1.0, "POS": 1.0, "NET_BANKING": .85, "IMPS": .5, "NEFT": .4, "BRANCH": .3}
HOURS = {"UPI": (7, 23), "ATM": (8, 22), "POS": (9, 22), "NET_BANKING": (6, 23), "IMPS": (6, 23),
         "NEFT": (8, 19), "BRANCH": (10, 16)}


def rrn12(dates, hh):
    return (z(dates.dt.year.values % 10, 1) + z(dates.dt.dayofyear.values, 3) + z(hh, 2) +
            z(rng.integers(0, 10 ** 6, len(dates)), 6)).astype(object)


def hhmmss(hh, mm_=None):
    n = len(hh)
    return (z(hh, 2) + ":" + z(rng.integers(0, 60, n), 2) + ":" + z(rng.integers(0, 60, n), 2)).astype(object)


def build_narration(ch, dr, rrn, pname, merch, city, biller, utr, chq, hd):
    out = []
    for c, d, rn, pn, mc, ct, bl, ut, cq, h_ in zip(ch, dr, rrn, pname, merch, city, biller, utr, chq, hd):
        if c == "UPI":
            if d and mc:
                out.append(f"UPI/DR/{rn}/{mc}/{mc.lower().replace(' ', '')}@{h_}")
            else:
                out.append(f"UPI/{'DR' if d else 'CR'}/{rn}/{pn}/{pn.split()[0].lower()}@{h_}")
        elif c == "ATM":
            out.append(f"ATM WDL/{BANK_CODE[:2]}{ut[-6:]}/{ct}")
        elif c == "POS":
            out.append(f"POS/{mc}/{ct}")
        elif c == "NET_BANKING":
            out.append(f"BILLPAY/{bl}/{ut[-10:]}" if d else f"NET TRF/{rn}/{pn}")
        elif c == "IMPS":
            out.append(f"IMPS/{'DR' if d else 'CR'}/{rn}/{pn}")
        elif c == "NEFT":
            out.append(f"NEFT/{'DR' if d else 'CR'}/{ut}/{pn}")
        else:
            out.append(f"CHQ WDL/{cq}" if d else f"CASH DEP/{cq}")
    return np.array(out, dtype=object)


# ---- random day-to-day transactions
el = acct[(acct.status == "ACTIVE") & acct.account_type.isin(["SAVINGS", "SALARY", "CURRENT"])]
wa = rng.lognormal(0, 0.9, len(el))
wa[el.account_type.values == "CURRENT"] *= 2
wa /= wa.sum()
NR = N_TXN_RANDOM
r_acc = rng.choice(el.account_number.values, NR, p=wa)
dts = pd.date_range(START, END)
dwt = np.where(dts.dayofweek >= 5, 0.7, 1.0)
dwt = dwt / dwt.sum()
r_date = pd.Series(dts[rng.choice(len(dts), NR, p=dwt)])
opn = pd.Series(r_acc).map(acct.set_index("account_number").open_date)
r_date = r_date.where(r_date > opn, opn + DAY)
r_ch = rng.choice(CH, NR, p=CH_P).astype(object)
r_dr = rng.random(NR) < np.array([P_DEBIT[c] for c in r_ch])
lo_h = np.array([HOURS[c][0] for c in r_ch])
hi_h = np.array([HOURS[c][1] for c in r_ch])
r_hh = lo_h + (rng.random(NR) * (hi_h - lo_h)).astype(int)
a = np.zeros(NR)
for c, (kind, p1, p2, lo_, hi_) in {"UPI": ("ln", 5.9, 1.1, 10, 100000), "POS": ("ln", 6.9, 1.0, 50, 200000),
                                     "NET_BANKING": ("ln", 7.6, 1.1, 100, 500000), "IMPS": ("ln", 8.3, 1.2, 100, 500000),
                                     "NEFT": ("ln", 9.4, 1.3, 1000, 2000000)}.items():
    m_ = r_ch == c
    a[m_] = np.clip(rng.lognormal(p1, p2, m_.sum()), lo_, hi_)
m_ = r_ch == "ATM"
a[m_] = rng.choice([500, 1000, 2000, 3000, 5000, 10000, 20000], m_.sum(), p=[.14, .20, .25, .10, .16, .10, .05])
m_ = r_ch == "BRANCH"
a[m_] = np.maximum(500, np.round(rng.lognormal(9.6, .9, m_.sum()) / 500) * 500)
a = np.where(rng.random(NR) < 0.7, np.round(a), np.round(a, 2))
a = np.where(np.isin(r_ch, ["ATM", "BRANCH"]), np.round(a), a)
# anomaly days: average transaction value spikes (kept consistent with running balance)
spike_days = rng.choice(dts[(dts.dayofweek < 5)][20:-3], 5, replace=False) if INJECT_ERRORS else []
sp_orig = np.full(NR, np.nan)
if INJECT_ERRORS:
    sm_ = r_date.isin(spike_days).values & (rng.random(NR) < 0.6) & ~np.isin(r_ch, ["ATM"])
    sp_orig[sm_] = a[sm_]
    a[sm_] = a[sm_] * 10
keep = (r_date <= END).values
rrn_ = rrn12(r_date, r_hh)
pname = np.array([f"{x} {y}".upper() for x, y in zip(rng.choice(ALL_FIRST, NR), rng.choice(ALL_LAST, NR))], dtype=object)
merch = np.where(r_ch == "UPI", np.where(rng.random(NR) < 0.6, rng.choice(UPI_M, NR), ""), rng.choice(POS_M, NR)).astype(object)
narr = build_narration(r_ch, r_dr, rrn_, pname, merch, rng.choice(CITIES_UP, NR), rng.choice(BILLERS, NR),
                       ("N" + z(rng.integers(0, 10 ** 15, NR), 15)), z(rng.integers(0, 10 ** 6, NR), 6),
                       rng.choice(HANDLES, NR))
utr_pref = rng.choice(BANKS4, NR)
narr = np.array([s.replace("NEFT/DR/N", f"NEFT/DR/{p}N").replace("NEFT/CR/N", f"NEFT/CR/{p}N") for s, p in zip(narr, utr_pref)], dtype=object)
t_rand = pd.DataFrame({"account_number": r_acc, "txn_date": r_date, "txn_time": hhmmss(r_hh),
                       "txn_type": np.where(r_dr, "DEBIT", "CREDIT").astype(object), "amount": a,
                       "channel": r_ch, "narration": narr, "_sp": sp_orig})[keep]

# ---- monthly salary credits
parts = [t_rand]
sa = acct[(acct.account_type == "SALARY") & (acct.status == "ACTIVE")].copy()
inc_map = cust.set_index("customer_id")["annual_income"]
sinc = sa.customer_id.map(inc_map).fillna(0).values
s_amt = np.round(np.maximum(sinc, 180000) / 12 * rng.uniform(0.78, 0.9, len(sa)), -2)
s_day = rng.integers(1, 6, len(sa))
s_emp = rng.choice(EMPLOYERS, len(sa))
s_utr = rng.choice(BANKS4, len(sa))
for ms in pd.date_range(START.replace(day=1), END, freq="MS"):
    due = ms + pd.to_timedelta(s_day - 1, unit="D")
    ok = (due >= START) & (due <= END) & (due > sa.open_date.values)
    k = int(ok.sum())
    if not k:
        continue
    hh = 9 + rng.integers(0, 3, k)
    parts.append(pd.DataFrame({
        "account_number": sa.account_number.values[ok], "txn_date": due[ok], "txn_time": hhmmss(hh),
        "txn_type": "CREDIT", "amount": s_amt[ok], "channel": "NEFT",
        "narration": [f"NEFT/CR/{u_}N{z(rng.integers(0, 10 ** 11), 11)}/SALARY {e}" for u_, e in zip(s_utr[ok], s_emp[ok])],
        "_sp": np.nan}))

# ---- monthly EMI debits
ln = loan[(loan.overdue_days == 0) & loan.status.isin(["ACTIVE", "CLOSED"])].copy()
ln_dm = ln.disbursement_date.dt.year.values * 12 + ln.disbursement_date.dt.month.values - 1
for ms in pd.date_range(START.replace(day=1), END, freq="MS"):
    due = ms + pd.to_timedelta(ln._emi_day.values - 1, unit="D")
    ok = (due >= START) & (due <= END) & (ms.year * 12 + ms.month - 1 >= ln_dm + 1) & (due <= ln.maturity_date.values)
    k = int(ok.sum())
    if not k:
        continue
    parts.append(pd.DataFrame({
        "account_number": ln.emi_account_number.values[ok], "txn_date": due[ok],
        "txn_time": hhmmss(4 + rng.integers(0, 2, k)), "txn_type": "DEBIT", "amount": ln.emi_amount.values[ok],
        "channel": "ACH", "narration": ["ACH DR/EMI/" + x for x in ln.loan_account_no.values[ok]], "_sp": np.nan}))

txn = pd.concat(parts, ignore_index=True)
txn = txn.sort_values(["txn_date", "txn_time"], kind="stable").reset_index(drop=True)
txn["txn_id"] = ("T" + txn.txn_date.dt.strftime("%Y%m%d") + (txn.groupby("txn_date").cumcount() + 1).astype(str).str.zfill(7)).astype(object)
if INJECT_ERRORS:
    sp = txn["_sp"].notna().values
    log_rows("transactions", txn.txn_id.values[sp], "amount", "AMOUNT_SPIKE", "ACCURACY", txn["_sp"][sp], txn["amount"][sp],
             [""] * int(sp.sum()), [""] * int(sp.sum()),
             "daily_average_spike_10x:" + ";".join(sorted(pd.Timestamp(d).strftime("%Y%m%d") for d in spike_days)))

# ---- running balance that reconciles with account balance
txn = txn.sort_values(["account_number", "txn_date", "txn_time", "txn_id"], kind="stable").reset_index(drop=True)
txn["_signed"] = np.where(txn.txn_type == "CREDIT", txn.amount, -txn.amount)
txn["_cum"] = txn.groupby("account_number")["_signed"].cumsum()
mn = txn.groupby("account_number")["_cum"].transform("min")
base_map = acct.set_index("account_number")["balance"]
buf_map = pd.Series(rng.uniform(300, 5000, len(acct)), index=acct.account_number)
opening = np.maximum(txn.account_number.map(base_map).values, -mn.values + txn.account_number.map(buf_map).values)
txn["balance_after"] = np.round(opening + txn["_cum"].values, 2)
final_bal = txn.groupby("account_number")["balance_after"].last()
acct["balance"] = acct.account_number.map(final_bal).fillna(acct.balance)
txn = txn.sort_values("txn_id", kind="stable").reset_index(drop=True)
txn = txn[["txn_id", "account_number", "txn_date", "txn_time", "txn_type", "amount", "balance_after", "channel", "narration"]]
objcols(txn)


# =====================================================================
#                 OPTIONAL: DELIBERATE DATA-QUALITY ERRORS
# =====================================================================
def run_injection(cust, acct, loan, txn, branch):
    nullf = lambda cur, idx: [None] * len(idx)
    T, ID = "branch_master", "branch_code"
    inject(branch, T, ID, "ifsc", 0.04, "INVALID_IFSC", "VALIDITY",
           lambda cur, idx: cur.map(lambda x: x[:10] if rng.random() < 0.5 else x.replace(BANK_CODE + "0", BANK_CODE + "1")))

    # ---------------- customers
    T, ID = "customer_master", "customer_id"
    n = len(cust)
    fd1, fd2 = END - 45 * DAY, END - 20 * DAY
    f1 = ((cust.source_system == "BRANCH_PORTAL") & (cust.load_date == fd1)).values
    f2 = ((cust.source_system == "MOBILE_APP") & (cust.load_date == fd2)).values
    n1 = f"upstream_feed_failure:BRANCH_PORTAL-{fd1:%Y%m%d}"
    n2 = f"upstream_feed_failure:MOBILE_APP-{fd2:%Y%m%d}"
    inject(cust, T, ID, "pin_code", None, "NULL_PIN", "COMPLETENESS", nullf, mask=f1 & (rng.random(n) < 0.7), notes=n1)
    inject(cust, T, ID, "mobile", None, "NULL_MOBILE", "COMPLETENESS", nullf, mask=f1 & (rng.random(n) < 0.7), notes=n1)
    inject(cust, T, ID, "mobile", None, "MOBILE_91_PREFIX", "VALIDITY", lambda cur, idx: "91" + cur,
           mask=f2 & (rng.random(n) < 0.65), notes=n2)
    inject(cust, T, ID, "mobile", 0.02, "NULL_MOBILE", "COMPLETENESS", nullf, "null")
    inject(cust, T, ID, "dob", 0.01, "NULL_DOB", "COMPLETENESS",
           lambda cur, idx: pd.Series([pd.NaT] * len(idx), dtype="datetime64[ns]"), "null")
    inject(cust, T, ID, "pin_code", 0.02, "NULL_PIN", "COMPLETENESS", nullf, "null")
    inject(cust, T, ID, "email", 0.02, "NULL_EMAIL", "COMPLETENESS", nullf, "null")
    inject(cust, T, ID, "pan", 0.02, "NULL_PAN", "COMPLETENESS", nullf, "pan")
    inject(cust, T, ID, "address_line1", 0.01, "NULL_ADDRESS", "COMPLETENESS", nullf, "null")
    inject(cust, T, ID, "city", 0.01, "NULL_CITY", "COMPLETENESS", nullf, "null")
    inject(cust, T, ID, "mobile", 0.010, "MOBILE_91_PREFIX", "VALIDITY", lambda cur, idx: "91" + cur, "mobile")
    inject(cust, T, ID, "mobile", 0.010, "MOBILE_9_DIGITS", "VALIDITY", lambda cur, idx: cur.str[:9], "mobile")
    inject(cust, T, ID, "mobile", 0.007, "MOBILE_LEADING_ZERO", "VALIDITY", lambda cur, idx: "0" + cur, "mobile")
    inject(cust, T, ID, "mobile", 0.005, "MOBILE_HAS_SPACE", "VALIDITY", lambda cur, idx: cur.str[:5] + " " + cur.str[5:], "mobile")
    inject(cust, T, ID, "mobile", 0.0015, "MOBILE_JUNK", "VALIDITY",
           lambda cur, idx: pd.Series(rng.choice(["9999999999", "0000000000", "1234567890", "9876543210"], len(idx))))
    inject(cust, T, ID, "pin_code", 0.008, "PIN_5_DIGITS", "VALIDITY", lambda cur, idx: cur.str[:5], "text")
    inject(cust, T, ID, "pin_code", 0.007, "PIN_7_DIGITS", "VALIDITY",
           lambda cur, idx: cur + pd.Series(rng.integers(0, 10, len(idx)).astype(str), index=cur.index), "text")

    def pin_other_city(cur, idx):
        cities = cust["city"].iloc[idx].fillna("").values
        j = rng.integers(0, NB, len(idx))
        for _ in range(20):
            same = pin_ref.city.values[j] == cities
            if not same.any():
                break
            j[same] = rng.integers(0, NB, int(same.sum()))
        return pin_ref.pin_code.values[j]

    inject(cust, T, ID, "pin_code", 0.007, "PIN_CITY_MISMATCH", "CONSISTENCY", pin_other_city)
    inject(cust, T, ID, "dob", 0.003, "DOB_IN_FUTURE", "VALIDITY",
           lambda cur, idx: pd.Series(END + pd.to_timedelta(rng.integers(30, 1500, len(idx)), unit="D")))
    inject(cust, T, ID, "dob", 0.001, "DOB_IMPLAUSIBLE", "VALIDITY",
           lambda cur, idx: pd.Series(pd.Timestamp("1850-01-01") + pd.to_timedelta(rng.integers(0, 18000, len(idx)), unit="D")))

    def bad_email(e):
        r_ = rng.random()
        if r_ < 0.30:
            return e.replace("@", "")
        if r_ < 0.55:
            return e.replace(".com", ".con") if ".com" in e else e + ".."
        if r_ < 0.75:
            return e.replace("gmail", "gmial") if "gmail" in e else e.replace("@", "@@")
        return e.replace("@", " @")

    def bad_pan(p):
        r_ = rng.random()
        if r_ < 0.30:
            return p.lower()
        if r_ < 0.55:
            return p[:9]
        if r_ < 0.80:
            return p[:2] + "1" + p[3:]
        return p[:5] + "ABCD" + p[9]

    inject(cust, T, ID, "email", 0.015, "EMAIL_INVALID", "VALIDITY", lambda cur, idx: cur.map(bad_email), "text")
    inject(cust, T, ID, "pan", 0.015, "PAN_INVALID_FORMAT", "VALIDITY", lambda cur, idx: cur.map(bad_pan), "pan")
    GV = {"M": ["Male", "male", "MALE", "m"], "F": ["Female", "female", "FEMALE", "f"]}
    inject(cust, T, ID, "gender", 0.04, "GENDER_NONSTANDARD", "CONSISTENCY",
           lambda cur, idx: cur.map(lambda g: rng.choice(GV[g])), "text")
    inject(cust, T, ID, "gender", 0.003, "GENDER_NAME_MISMATCH", "ACCURACY", lambda cur, idx: cur.map({"M": "F", "F": "M"}))

    def bad_name(nm):
        r_ = rng.random()
        return nm.upper() if r_ < .35 else nm.lower() if r_ < .60 else nm.replace(" ", "  ") if r_ < .85 else " " + nm

    inject(cust, T, ID, "full_name", 0.02, "NAME_FORMAT", "CONSISTENCY", lambda cur, idx: cur.map(bad_name), "text")
    inject(cust, T, ID, "full_name", 0.002, "NAME_INVALID_CHARS", "VALIDITY",
           lambda cur, idx: cur + pd.Series(rng.choice(["1", "2", "@", "#", "3"], len(idx)), index=cur.index))

    # ---------------- accounts
    T, ID = "account_master", "account_number"
    n = len(acct)
    is_sav = acct.account_type.isin(["SAVINGS", "SALARY"]).values
    inject(acct, T, ID, "balance", 0.0, "NEGATIVE_BALANCE_SAVINGS", "VALIDITY",
           lambda cur, idx: pd.Series(-np.round(rng.uniform(100, 50000, len(idx)), 2)), mask=is_sav & (rng.random(n) < 0.005))
    inject(acct, T, ID, "balance", 0.0, "CLOSED_ACCOUNT_WITH_BALANCE", "CONSISTENCY",
           lambda cur, idx: pd.Series(np.round(rng.uniform(1000, 100000, len(idx)), 2)),
           mask=(acct.status == "CLOSED").values & (rng.random(n) < 0.15))
    inject(acct, T, ID, "balance", 0.003, "NULL_BALANCE", "COMPLETENESS", lambda cur, idx: pd.Series([np.nan] * len(idx)), "null")
    inject(acct, T, ID, "customer_id", 0.004, "ORPHAN_CUSTOMER_ID", "INTEGRITY",
           lambda cur, idx: pd.Series((9_000_000_000 + rng.integers(0, 10 ** 9, len(idx))).astype(str)), "text")
    inject(acct, T, ID, "open_date", 0.002, "OPEN_DATE_IN_FUTURE", "VALIDITY",
           lambda cur, idx: pd.Series(END + pd.to_timedelta(rng.integers(10, 400, len(idx)), unit="D")))
    AT = {"SAVINGS": ["Savings", "SAV", "savings", "SB"], "SALARY": ["Salary", "SAL", "salary"],
          "CURRENT": ["Current", "CA", "current"], "FIXED_DEPOSIT": ["FD", "Fixed Deposit", "fixed_deposit"]}
    inject(acct, T, ID, "account_type", 0.015, "ACCOUNT_TYPE_NONSTANDARD", "CONSISTENCY",
           lambda cur, idx: cur.map(lambda x: rng.choice(AT[x])), "text")
    inject(acct, T, ID, "branch_code", 0.002, "INVALID_BRANCH_CODE", "INTEGRITY",
           lambda cur, idx: pd.Series([f"9{x:03d}" for x in rng.integers(0, 1000, len(idx))]))

    # ---------------- loans
    T, ID = "loan_accounts", "loan_account_no"
    n = len(loan)
    sdays = rng.choice(pd.date_range(START + 20 * DAY, END - 3 * DAY), 6, replace=False)
    inject(loan, T, ID, "disbursement_amount", 0.0, "DISBURSEMENT_SPIKE", "ACCURACY", lambda cur, idx: cur * 8,
           mask=loan.disbursement_date.isin(sdays).values & (rng.random(n) < 0.85),
           notes="daily_average_spike_8x:" + ";".join(sorted(pd.Timestamp(d).strftime("%Y%m%d") for d in sdays)))
    fd3 = END - 30 * DAY
    ff = ((loan.source_system == "LOAN_ORIGINATION") & (loan.load_date == fd3)).values
    early = lambda cur, idx: pd.Series(loan["disbursement_date"].iloc[idx].values - pd.to_timedelta(rng.integers(1, 365, len(idx)), unit="D"))
    inject(loan, T, ID, "maturity_date", 0.0, "MATURITY_BEFORE_DISBURSEMENT", "CONSISTENCY", early,
           mask=ff & (rng.random(n) < 0.5), notes=f"upstream_feed_failure:LOAN_ORIGINATION-{fd3:%Y%m%d}")
    inject(loan, T, ID, "maturity_date", 0.003, "MATURITY_BEFORE_DISBURSEMENT", "CONSISTENCY", early)
    inject(loan, T, ID, "maturity_date", 0.004, "MATURITY_OVER_30_YEARS", "VALIDITY",
           lambda cur, idx: pd.Series(loan["disbursement_date"].iloc[idx].values + pd.to_timedelta(rng.integers(31 * 365, 45 * 365, len(idx)), unit="D")))
    inject(loan, T, ID, "maturity_date", 0.003, "NULL_MATURITY_DATE", "COMPLETENESS",
           lambda cur, idx: pd.Series([pd.NaT] * len(idx), dtype="datetime64[ns]"), "null")
    inject(loan, T, ID, "customer_id", 0.008, "ORPHAN_CUSTOMER_ID", "INTEGRITY",
           lambda cur, idx: pd.Series((9_000_000_000 + rng.integers(0, 10 ** 9, len(idx))).astype(str)))
    inject(loan, T, ID, "disbursement_amount", 0.001, "NEGATIVE_AMOUNT", "VALIDITY", lambda cur, idx: -cur)
    inject(loan, T, ID, "disbursement_amount", 0.002, "NULL_AMOUNT", "COMPLETENESS", lambda cur, idx: pd.Series([np.nan] * len(idx)), "null")
    inject(loan, T, ID, "interest_rate", 0.003, "INTEREST_RATE_OUT_OF_RANGE", "VALIDITY",
           lambda cur, idx: pd.Series(rng.choice([0.0, 0.5, 42.5, 120.0], len(idx))))
    inject(loan, T, ID, "disbursement_date", 0.002, "DISBURSEMENT_DATE_IN_FUTURE", "VALIDITY",
           lambda cur, idx: pd.Series(END + pd.to_timedelta(rng.integers(10, 400, len(idx)), unit="D")))

    # ---------------- transactions
    T, ID = "transactions", "txn_id"
    inject(txn, T, ID, "account_number", 0.0015, "ORPHAN_ACCOUNT_NUMBER", "INTEGRITY",
           lambda cur, idx: pd.Series([f"9{x:03d}01{y:06d}" for x, y in zip(rng.integers(0, 1000, len(idx)), rng.integers(0, 10 ** 6, len(idx)))]))
    inject(txn, T, ID, "txn_date", 0.0005, "TXN_DATE_IN_FUTURE", "VALIDITY",
           lambda cur, idx: pd.Series(END + pd.to_timedelta(rng.integers(5, 300, len(idx)), unit="D")))
    nd_t = int(len(txn) * 0.001)
    di_ = rng.choice(len(txn), nd_t, replace=False)
    dt = txn.iloc[di_].copy()
    dt["txn_id"] = [f"T{END:%Y%m%d}{9_000_000 + i:07d}" for i in range(nd_t)]
    log_rows(T, dt["txn_id"].values, "txn_id", "DUPLICATE_TXN", "UNIQUENESS", txn["txn_id"].iloc[di_].values,
             dt["txn_id"].values, [""] * nd_t, [""] * nd_t, [f"duplicate_of={x}" for x in txn["txn_id"].iloc[di_].values])
    return pd.concat([txn, dt], ignore_index=True)


if INJECT_ERRORS:
    txn = run_injection(cust, acct, loan, txn, branch)

# =====================================================================
#                         BATCH LOG + CORRECTION HISTORY
# =====================================================================
parts = []
for df, tb in [(cust, "customer_master"), (acct, "account_master"), (loan, "loan_accounts")]:
    g = df.groupby(["batch_id", "source_system", "load_date"]).size().reset_index(name="row_count")
    g["table_name"] = tb
    parts.append(g)
tg = txn[(txn.txn_date >= START) & (txn.txn_date <= END)].groupby("txn_date").size().reset_index(name="row_count")
tg["load_date"], tg["source_system"], tg["table_name"] = tg["txn_date"], "CBS", "transactions"
tg["batch_id"] = "CBS-" + tg["load_date"].dt.strftime("%Y%m%d")
parts.append(tg[["batch_id", "source_system", "load_date", "row_count", "table_name"]])
blog = pd.concat(parts, ignore_index=True).sort_values(["load_date", "batch_id", "table_name"]).reset_index(drop=True)
blog["start_time"] = pd.to_datetime(rng.integers(3600, 18000, len(blog)), unit="s").strftime("%H:%M:%S")
blog["duration_sec"] = np.round(blog.row_count * rng.uniform(0.01, 0.04, len(blog)) + rng.uniform(5, 40, len(blog))).astype(int)
blog["status"] = "COMPLETED"
blog = blog[["batch_id", "source_system", "table_name", "load_date", "start_time", "duration_sec", "row_count", "status"]]


def build_history(total):
    GV = {"M": ["Male", "male", "MALE", "m"], "F": ["Female", "female", "FEMALE", "f"]}
    spec = [("MOBILE_91_PREFIX", "mobile", "RULE_BASED", .22, .985), ("MOBILE_9_DIGITS", "mobile", "RULE_BASED", .10, .40),
            ("GENDER_NONSTANDARD", "gender", "RULE_BASED", .18, .985), ("NAME_FORMAT", "full_name", "RULE_BASED", .12, .97),
            ("PAN_LOWERCASE", "pan", "RULE_BASED", .05, .99), ("PIN_5_DIGITS", "pin_code", "ML_MODEL", .10, .70),
            ("EMAIL_INVALID", "email", "ML_MODEL", .08, .90), ("DUPLICATE_NAME_VARIANT", "full_name", "CLUSTERING", .15, .75)]
    counts = rng.multinomial(total, [s[3] for s in spec])
    NOTE = {"MOBILE_9_DIGITS": "Mobile cannot start with 0 - verify with customer", "PIN_5_DIGITS": "Recommended PIN not in customer city",
            "EMAIL_INVALID": "Domain suggestion incorrect", "DUPLICATE_NAME_VARIANT": "Different person - DOB and PAN do not match"}
    recs = []
    for (et, col, meth, _, ap), k in zip(spec, counts):
        for _ in range(int(k)):
            f = str(rng.choice(ALL_FIRST)); l = str(rng.choice(ALL_LAST)); nm = f"{f} {l}"
            mob = str(gen_mobiles(1)[0])
            if et == "MOBILE_91_PREFIX":
                wrong, rec, conf = "91" + mob, mob, rng.uniform(.90, .99)
            elif et == "MOBILE_9_DIGITS":
                wrong = mob[:9]; rec = "0" + wrong; conf = rng.uniform(.60, .85)
            elif et == "GENDER_NONSTANDARD":
                g = str(rng.choice(["M", "F"])); wrong = str(rng.choice(GV[g])); rec = g; conf = rng.uniform(.92, .99)
            elif et == "NAME_FORMAT":
                wrong = nm.upper() if rng.random() < .6 else nm.lower(); rec = nm; conf = rng.uniform(.90, .99)
            elif et == "PAN_LOWERCASE":
                p = str(gen_pan([l])[0]); wrong, rec, conf = p.lower(), p, rng.uniform(.95, .99)
            elif et == "PIN_5_DIGITS":
                p = str(pin_ref.pin_code.values[rng.integers(0, NB)]); wrong, rec, conf = p[:5], p, rng.uniform(.55, .95)
            elif et == "EMAIL_INVALID":
                e = f"{f.lower()}.{l.lower()}{rng.integers(1, 99)}@gmail.com"
                wrong = e.replace("gmail", "gmial") if rng.random() < .5 else e.replace(".com", ".con"); rec, conf = e, rng.uniform(.60, .95)
            else:
                wrong, rec, conf = name_variant(nm), nm, rng.uniform(.60, .95)
            approved = rng.random() < min(.995, max(.05, ap + (conf - .75) * (.6 if meth != "RULE_BASED" else .1)))
            if not approved and meth != "RULE_BASED" and et in NOTE:
                rec = {"PIN_5_DIGITS": str(pin_ref.pin_code.values[rng.integers(0, NB)]), "EMAIL_INVALID": wrong.replace("gmial", "gmail.co"),
                       "DUPLICATE_NAME_VARIANT": f"{rng.choice(ALL_FIRST)} {l}"}[et]
            recs.append(("customer_master", col, et, wrong, rec, meth, round(conf, 3), "APPROVED" if approved else "REJECTED",
                         "" if approved else NOTE.get(et, "Rejected by steward")))
    h = pd.DataFrame(recs, columns=["table_name", "column_name", "error_type", "wrong_value", "recommended_value",
                                    "correction_method", "confidence_score", "decision", "reviewer_notes"])
    h = h.sample(frac=1, random_state=SEED).reset_index(drop=True)
    h.insert(0, "correction_id", [f"H{i + 1:06d}" for i in range(len(h))])
    h["reviewed_by"] = rng.choice(["DS_ANITA", "DS_VIKAS", "DS_SANA", "DS_ROHIT", "DS_MEERA"], len(h))
    h["reviewed_on"] = rand_dates(len(h), pd.Timestamp("2025-10-01"), pd.Timestamp("2026-03-20"))
    return h[["correction_id", "table_name", "column_name", "error_type", "wrong_value", "recommended_value",
              "correction_method", "confidence_score", "decision", "reviewed_by", "reviewed_on", "reviewer_notes"]]


hist = build_history(N_HISTORY)

# =====================================================================
#                                 SAVE
# =====================================================================
def save(df, name, sort_col=None):
    out = df.sort_values(sort_col).reset_index(drop=True) if sort_col else df.copy()
    out = out[[c for c in out.columns if not c.startswith("_")]]
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d").fillna("")
    out.to_csv(os.path.join(OUT, name + ".csv"), index=False, float_format="%.2f")
    print(f"{name}.csv  rows={len(out):,}")


save(pin_ref, "pincode_reference")
save(branch, "branch_master", "branch_code")
save(cust, "customer_master", "customer_id")
save(acct, "account_master", "account_number")
save(loan, "loan_accounts", "loan_account_no")
save(txn, "transactions", "txn_id")
save(blog, "batch_log")
save(hist, "correction_history")
if LOG:
    err = pd.concat(LOG, ignore_index=True).sort_values(["table_name", "record_id", "column_name"]).reset_index(drop=True)
    err.insert(0, "error_id", [f"E{i + 1:07d}" for i in range(len(err))])
    save(err, "error_log_ground_truth")

DDL = """-- Oracle DDL for the realistic bank dataset. Run with F5 (Run Script), then import each CSV.
-- Import settings: Header ticked, Insert method, date format YYYY-MM-DD. Values contain no commas or quotes.
-- No foreign keys on purpose (the data may contain orphan records that a DQ tool should find).
CREATE TABLE pincode_reference (pin_code VARCHAR2(10), area_name VARCHAR2(60), city VARCHAR2(60), state VARCHAR2(60),
    CONSTRAINT pk_pincode_reference PRIMARY KEY (pin_code, area_name));
CREATE TABLE branch_master (branch_code VARCHAR2(6) PRIMARY KEY, branch_name VARCHAR2(80), ifsc VARCHAR2(15),
    micr_code VARCHAR2(12), address VARCHAR2(150), city VARCHAR2(60), state VARCHAR2(60), pin_code VARCHAR2(10), opened_date DATE);
CREATE TABLE customer_master (customer_id VARCHAR2(12) PRIMARY KEY, full_name VARCHAR2(100), dob DATE, gender VARCHAR2(10),
    marital_status VARCHAR2(12), mobile VARCHAR2(20), email VARCHAR2(120), address_line1 VARCHAR2(120), address_line2 VARCHAR2(80),
    city VARCHAR2(60), state VARCHAR2(60), pin_code VARCHAR2(10), pan VARCHAR2(15), aadhaar_last4 VARCHAR2(6),
    occupation VARCHAR2(20), annual_income NUMBER(14), kyc_status VARCHAR2(12), kyc_date DATE, customer_since DATE,
    home_branch_code VARCHAR2(6), source_system VARCHAR2(30), load_date DATE, batch_id VARCHAR2(40));
CREATE TABLE account_master (account_number VARCHAR2(16) PRIMARY KEY, customer_id VARCHAR2(12), account_type VARCHAR2(20),
    branch_code VARCHAR2(6), ifsc VARCHAR2(15), open_date DATE, balance NUMBER(18,2), currency VARCHAR2(3), status VARCHAR2(12),
    source_system VARCHAR2(30), load_date DATE, batch_id VARCHAR2(40));
CREATE TABLE loan_accounts (loan_account_no VARCHAR2(16) PRIMARY KEY, customer_id VARCHAR2(12), loan_type VARCHAR2(20),
    branch_code VARCHAR2(6), disbursement_date DATE, disbursement_amount NUMBER(18,2), tenure_months NUMBER(4),
    interest_rate NUMBER(6,2), emi_amount NUMBER(14,2), emi_account_number VARCHAR2(16), maturity_date DATE,
    outstanding_principal NUMBER(18,2), overdue_days NUMBER(5), status VARCHAR2(12), source_system VARCHAR2(30),
    load_date DATE, batch_id VARCHAR2(40));
CREATE TABLE transactions (txn_id VARCHAR2(20) PRIMARY KEY, account_number VARCHAR2(16), txn_date DATE, txn_time VARCHAR2(8),
    txn_type VARCHAR2(8), amount NUMBER(18,2), balance_after NUMBER(18,2), channel VARCHAR2(15), narration VARCHAR2(150));
CREATE TABLE batch_log (batch_id VARCHAR2(40), source_system VARCHAR2(30), table_name VARCHAR2(30), load_date DATE,
    start_time VARCHAR2(8), duration_sec NUMBER(8), row_count NUMBER(10), status VARCHAR2(20),
    CONSTRAINT pk_batch_log PRIMARY KEY (batch_id, table_name));
CREATE TABLE correction_history (correction_id VARCHAR2(10) PRIMARY KEY, table_name VARCHAR2(30), column_name VARCHAR2(30),
    error_type VARCHAR2(50), wrong_value VARCHAR2(200), recommended_value VARCHAR2(200), correction_method VARCHAR2(20),
    confidence_score NUMBER(4,3), decision VARCHAR2(10), reviewed_by VARCHAR2(30), reviewed_on DATE, reviewer_notes VARCHAR2(200));
CREATE TABLE error_log_ground_truth (error_id VARCHAR2(10) PRIMARY KEY, table_name VARCHAR2(30), record_id VARCHAR2(24),
    column_name VARCHAR2(30), error_type VARCHAR2(50), dq_dimension VARCHAR2(20), original_value VARCHAR2(200),
    injected_value VARCHAR2(200), source_system VARCHAR2(30), batch_id VARCHAR2(40), notes VARCHAR2(200));
"""
with open(os.path.join(OUT, "01_create_tables_oracle.sql"), "w") as fh:
    fh.write(DDL)
print("01_create_tables_oracle.sql written")
