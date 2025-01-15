SeisAutoFaultID is a Python-based workflow designed for automatic fault identification using the spatial and temporal distribution of microseismic events. Developed and applied to the FORGE stimulation stages in 2022 and 2024, this workflow employs advanced clustering identify and characterize faults.
Features:
•	Clustering with DBSCAN: Groups seismic events based on time or distance density to identify clusters representing faults.
•	Fault Plane Fitting with PCA: Calculates key fault properties, including center, axes, length, width, dip, dip direction, and strike.
•	Dynamic Iterative Refinement: Refines fault clusters using proximity-based criteria to improve clustering accuracy.
•	Comprehensive Fault Properties: Outputs detailed fault attributes aligned with field observations.
Application:
This workflow is capable of identifying and characterizing faults in seismic datasets, with visualizations of spatial event distributions, and fitted fault planes. It has been validated with data from the FORGE stimulation stages and demonstrates high accuracy and reliability in fault detection.
