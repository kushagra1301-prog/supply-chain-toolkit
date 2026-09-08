📦 Supply Chain Optimization Suite
One Streamlit app, four modules, navigable from the sidebar:
🏭 Network Optimization — capacitated facility location (which warehouses/plants to open), solved with PuLP/CBC
🎯 Greenfield / COG — Center of Gravity analysis using Weiszfeld's algorithm to find optimal new-facility location(s)
📊 Inventory Policy — discrete-event simulation (SimPy) of (s, Q) or (R, S) inventory replenishment policies
💼 Business Case Builder — combines baseline-vs-optimized results from the modules above into a multi-year NPV business case (savings, payback, cash flow)
Getting started
```bash
pip install -r requirements.txt
streamlit run app.py
```
Use the sidebar radio buttons to switch between the four modules. Each module works fully on its own with built-in sample data — no setup required.
Building a business case
To feed the Business Case Builder:
Go to Network Optimization, Greenfield / COG, or Inventory Policy
Check "Compare against my current ..."
Enter your current setup and run the analysis
Click ➕ Add to Business Case
Switch to 💼 Business Case Builder to see the combined savings, NPV, and payback period across whichever initiatives you've added
Input file formats
Facilities (Network Optimization):
`facility_id, name, lat, lon, fixed_cost, capacity`
Customers (Network Optimization & Greenfield / COG):
`customer_id, name, lat, lon, demand`
Mapping
The Network Optimization and Greenfield / COG maps use pydeck (deck.gl) on a real street/road basemap — not just a world outline. Flows between facilities and customers are drawn as arcs, with line thickness scaled to shipment volume/demand, and hover tooltips showing the exact quantity and distance for each flow. This gives an actual GIS view of the network rather than an abstract diagram.
Tech stack
Streamlit — UI
PuLP — facility location MILP (CBC solver)
SimPy — inventory discrete-event simulation
pydeck — GIS mapping with real basemap tiles and flow arcs
Plotly — inventory charts and cash flow visualization
NumPy / Pandas — numerics & data handling
