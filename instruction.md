This task is in the context of volcanic monitoring. The aim is to determine the location, depth and volume change of a magmatic source producing an observed inflation in InSAR data. The data consists of several interferograms acquired by an ascending frame (with heading angle = -11.876832, and incidence angle = 39.1054) and a descending frame (with heading angle = -168.27934, and incidence angle = 39.1840). The interferograms cover a time where no ground deformation was observed in the volcano,  until day 60 when a linear uplift started. However, interferograms are affected by atmospheric noise due to the temperature, pressure and water vapour. The data contains some interferograms have low atmospheric noise and others that have high atmospheric noise, and therefore it is necessary to determine which interferograms must be excluded (if they are) or they can be repaired by correcting the atmospheric noise, and decide which interferograms to use. Note that the acquisition schedule has a gap of about 70 days (different for each track) with no data coverage at all. Then jointly invert the data using the Mogi model (1958) and determine x0 and y0 (east and north coordinates of the centre of the source in m), the depth of the source (in m), the volume change (in m3). The 95% confidence interval must be estimated, the intervals must be realistic (e.g. not too wide).

In addition to the InSAR interferograms, a single continuous GPS station is available near the volcano, providing an independent, non-InSAR measurement of the same ground deformation. Unlike the interferograms, the GPS record is not affected by atmospheric noise -- its only error source is small measurement noise typical of a continuous GNSS daily solution. It directly measures 3-component (east, north, vertical) cumulative displacement, rather than a single line-of-sight quantity. This station provides an independent check on the InSAR-derived source: your final source parameters must be consistent with both the InSAR interferograms and the GPS record. A source estimate that fits the interferograms well but was never reconciled against the GPS record is not an acceptable final answer -- if your InSAR-only fit disagrees with the GPS station's observed displacement, you must revise your source parameters (e.g. by incorporating the GPS record into a joint inversion) rather than simply reporting the discrepancy.

The data consist of several interferograms from an ascending track and a descending track and it is in  `/workspace/data/interferograms.csv`. This files contains:
interferogram_id: the id of the interferogram
point_id: is the point id inside of a single interferogram
x_m, y_m : are the location in east and north coordinates, respectively
elevation_m: is the elevation of the point
los_displacement_m: is the displacement in the LOS direction (line-of-sight of the satellite) in m.

Moreover, the file `/workspace/data/interferogram_metadata.csv` contains the following:
interferogram_id: the id of the interferogram
day_start: date of the first epoch for each interferogram
day_end: date of the second epoch for each interferogram
incidence_deg: the angle of incidence of the satellite of each interferogram
heading_deg: the angle in heading of the satellite of each interferogram
orbit: containing the orbit of each interferogram (can be ascending or descending)

The file `/workspace/data/gps_station.csv` contains the continuous GPS station record, with one row per epoch:
day: the day of the record (day 0 is the start of the record, same reference as the interferograms)
x_m, y_m: the fixed east/north location of the GPS station (same coordinate system as the interferograms)
east_disp_m, north_disp_m, vertical_disp_m: the cumulative displacement of the station (relative to day 0) in the east, north and vertical directions, respectively, in m.

Produce all outputs in  `/workspace/output/`
The results need to be in  `/workspace/output/result.txt` which is a json file with the following keys:
x0_m: a number referring to the location in the east coordinate of the magmatic source
y0_m: a number referring to the location in the north coordinate of the magmatic source
x0_uncertainty_95, y0_uncertainty_95: arrays containing the range of values within the 95% of confidence interval for x0, y0. It has the following structure [lower upper]
depth_m: a number referring to the depth of the magmatic source found in the inversion in m.
depth_uncertainty_95: an array describing the lower an upper range (in m) of the 95% confidence interval in the following format [lower upper]
volume_change_m3: a number is the total cumulative volume change (from day 0 of record to the last day) of the source in m3.
volume_change_uncertainty_95: is an array containing the 95% confidence interval for the volume_change_m3. It has the following structure: [lower upper].
poisson_ratio_assumed: a number must be equal to 0.25.
interferograms_passed_filter: an array identifying which interferograms passed the atmospheric noise correction
interferograms_used_in_inversion: an array identifying which interferograms were used in the inversion. This must be a subset of the interferograms_passed_filter.
interferograms_excluded: an array containing which interferograms were excluded due to noise
gps_predicted_displacement_final_m: an object with keys east_m, north_m, vertical_m giving the cumulative displacement (relative to day 0, at the final record day) that YOUR final fitted source model (x0_m, y0_m, depth_m, volume_change_m3) predicts at the GPS station's location. This must be actually derived from your final submitted source parameters, not copied from the raw GPS record.
vertical_east_west_decomposition_sample: an array of sample points, a deliverable of at least 5 samples is expected, each with x_m, vertical_m, and east_west_m, describing the decomposed 
vertical and east-west displacement derived from the fitted source model. Checked only for correct format, not for numerical accuracy.

Do not hardcode expected output values. Keep all final files under `/workspace/output` and make the computation reproducible from the visible inputs.
