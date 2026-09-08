# 📦 Supply Chain Optimization Suite

One Streamlit app, four modules, navigable from the sidebar:

1. **🏭 Network Optimization** — capacitated facility location (which warehouses/plants to open), solved with PuLP/CBC
2. **🎯 Greenfield / COG** — Center of Gravity analysis using Weiszfeld's algorithm to find optimal new-facility location(s)
3. **📊 Inventory Policy** — discrete-event simulation (SimPy) of (s, Q) or (R, S) inventory replenishment policies
4. **💼 Business Case Builder** — combines baseline-vs-optimized results from the modules above into a multi-year NPV business case (savings, payback, cash flow)

## Getting started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the sidebar radio buttons to switch between the four modules. Each module works fully on its own with built-in sample data — no setup required.

## Building a business case

To feed the Business Case Builder:
1. Go to **Network Optimization**, **Greenfield / COG**, or **Inventory Policy**
2. Check **"Compare against my current ..."**
3. Enter your current setup and run the analysis
4. Click **➕ Add to Business Case**
5. Switch to **💼 Business Case Builder** to see the combined savings, NPV, and payback period across whichever initiatives you've added

## Input file formats

**Facilities** (Network Optimization):
`facility_id, name, lat, lon, fixed_cost, capacity`

**Customers** (Network Optimization & Greenfield / COG):
`customer_id, name, lat, lon, demand`

## Tech stack
- [Streamlit](https://streamlit.io/) — UI
- [PuLP](https://coin-or.github.io/pulp/) — facility location MILP (CBC solver)
- [SimPy](https://simpy.readthedocs.io/) — inventory discrete-event simulation
- [Plotly](https://plotly.com/python/) — interactive maps and charts
- [NumPy](https://numpy.org/) / [Pandas](https://pandas.pydata.org/) — numerics & data handling
