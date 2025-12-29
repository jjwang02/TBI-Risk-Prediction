import pandas as pd
import numpy as np

def prepare_tbi_data(data: pd.DataFrame, 
                      replace_special_codes: bool = True,
                      check_consistency: bool = True,
                    **kwargs) -> pd.DataFrame:
  """
  Prepares traumatic brain injury (TBI) data by performing data cleaning and consistency checks.

  This function applies several preprocessing steps to a TBI dataset, including:

  - Replacing special codes (specifically '92') in certain columns based on the values of related columns.
  - Checking and correcting the consistency of GCS scores.
  - Allows for future extensions to handle other data transformations via keyword arguments.

  Args:
    data: A pandas DataFrame containing TBI data.  
    replace_special_codes: If True, replaces specific '92' codes with 0 or NaN based on the value of related parent columns. Defaults to True.
    check_consistency: If True, checks the consistency between 'GCSTotal' and the sum of 'GCSMotor', 'GCSVerbal', and 'GCSEye'.  Inconsistent 'GCSTotal' values are corrected, and 'GCSGroup' is re-calculated. Defaults to True.
    **kwargs:  Placeholder for potential future extensions.  Currently unused.

  Returns:
    A pandas DataFrame with the prepared TBI data.
  
  """

  data_copy = data.copy() # Important to avoid modifying the original DataFrame in place.

  if replace_special_codes:

    # Replace some of the 92 codes as 0 in the AMS sub-items, if AMS is answered as No. 
    # Then replace the rest of the 92 codes as NA.
    ams_is_zero = data_copy['AMS'] == 0
    cols = ['AMSAgitated', 'AMSSleep', 'AMSSlow', 'AMSRepeat', 'AMSOth']
    data_copy.loc[ams_is_zero, cols] = 0 
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the SFxBas sub-items, if SFxBas is answered as No. Rest of 92 would be NA.
    sf_is_zero = data_copy['SFxBas'] == 0
    cols = ['SFxBasHem', 'SFxBasOto', 'SFxBasPer', 'SFxBasRet', 'SFxBasRhi']
    data_copy.loc[sf_is_zero, cols] = 0 
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the Hema sub-items, if Hema is answered as No. Rest of 92 would be NA.
    hema_is_zero = data_copy['Hema'] == 0
    cols = ['HemaLoc', 'HemaSize']
    data_copy.loc[hema_is_zero, cols] = 0
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the Clav sub-items, if Clav is answered as No. Rest of 92 would be NA.
    cl_is_zero = data_copy['Clav'] == 0
    cols = ['ClavFace', 'ClavNeck', 'ClavFro', 'ClavOcc', 'ClavPar', 'ClavTem']
    data_copy.loc[cl_is_zero, cols] = 0
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the NeuroD sub-items, if NeuroD is answered as No. Rest of 92 would be NA.
    nr_is_zero = data_copy['NeuroD'] == 0
    cols = ['NeuroDMotor', 'NeuroDSensory', 'NeuroDCranial', 'NeuroDReflex', 'NeuroDOth']
    data_copy.loc[nr_is_zero, cols] = 0
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the OSI sub-items, if OSI is answered as No. Rest of 92 would be NA.
    osi_is_zero = data_copy['OSI'] == 0
    cols = ['OSIExtremity', 'OSICut', 'OSICspine', 'OSIFlank', 'OSIAbdomen', 'OSIPelvis', 'OSIOth']
    data_copy.loc[osi_is_zero, cols] = 0
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    # Replace some of the 92 codes as 0 in the CTForm1 sub-items, if CTForm1 is answered as No. Rest of 92 would be NA.
    ctf_is_zero = data_copy['CTForm1'] == 0
    cols = ['IndAge', 'IndAMS', 'IndClinSFx', 'IndHA', 'IndHema', 'IndLOC', 'IndMech', 'IndNeuroD','IndRqstMD','IndRqstParent','IndRqstTrauma','IndSeiz','IndVomit','IndXraySFx','IndOth']
    data_copy.loc[ctf_is_zero, cols] = 0
    data_copy[cols] = data_copy[cols].replace(92, np.nan)

    data_copy['PosCT'] = data_copy['PosCT'].replace(92, np.nan)
    data_copy['Amnesia_verb'] = data_copy['Amnesia_verb'].replace(91, np.nan)
    data_copy['HA_verb'] = data_copy['HA_verb'].replace(91, np.nan)



  if check_consistency:
    
    # Create a mask to identify rows with NA in GCSMotor, GCSVerbal, or GCSEye
    na_mask = data_copy[['GCSMotor', 'GCSVerbal', 'GCSEye']].isna().any(axis=1)

    data_copy['GCS_Sum'] = data_copy[['GCSMotor', 'GCSVerbal', 'GCSEye']].sum(axis=1)

    data_copy['GCSTotal_Correct'] = data_copy['GCSTotal'] == data_copy['GCS_Sum']

    data_copy['GCSTotal_Correct'] = data_copy['GCSTotal_Correct'].astype(float) 
    data_copy.loc[na_mask, 'GCSTotal_Correct'] = pd.NA


    incorrect_rows = data_copy[(data_copy['GCSTotal_Correct'] == False) & (~na_mask)]
    if not incorrect_rows.empty:
        print("The following rows have incorrect GCSTotal values (excluding rows with NAs):")
        print(incorrect_rows[['GCSMotor', 'GCSVerbal', 'GCSEye', 'GCSTotal', 'GCS_Sum']])
        
        # Replace with correct GCSTotal
        data_copy.loc[incorrect_rows.index, 'GCSTotal'] = data_copy.loc[incorrect_rows.index, 'GCS_Sum']
        print("\nGCSTotal values corrected for the incorrect rows.")
    else:
        if data_copy[~na_mask].empty:
            print("No rows to check because all contain NA in GCSMotor, GCSVerbal, or GCSEye")
        else:
            print("All rows have correct GCSTotal values (excluding rows with NAs).")

    # Remove the intermediate 'GCS_Sum' column
    data_copy.drop('GCS_Sum', axis=1, inplace=True)

    # Replace GCSGRroup with correct values
    data_copy['GCSGroup'] = np.nan
    not_na_mask = data_copy['GCSTotal'].notna()
    data_copy.loc[not_na_mask & (data_copy['GCSTotal'].between(3, 13)), 'GCSGroup'] = 1
    data_copy.loc[not_na_mask & (data_copy['GCSTotal'].between(14, 15)), 'GCSGroup'] = 2





  # Code for other action items would go here, potentially using kwargs.  For example:
  # if 'some_other_transformation' in kwargs:
  #   data_copy['some_other_column'] = kwargs['some_other_transformation'](data_copy['some_other_column'])

  return data_copy

