# main.py

import pandas as pd
from cleaning import prepare_tbi_data
from helper import plot_after_cleaning, plot_before_cleaning,tex_to_pdf


def main():
    """
    The main entry point of the program, loads data, preprocesses it, and visualizes it.
    """
    print("Starting main function...")

    print("1. Reading original data...")
    try:
        df_original = pd.read_csv('../data/TBI PUD 10-08-2013.csv')
        print("Original data read successfully.")
    except FileNotFoundError:
        print("Error:  '../data/TBI PUD 10-08-2013.csv' file not found. Please ensure the file path is correct.")
        return  # Exit the program
    except Exception as e:
        print(f"Error reading data: {e}")
        return  # Exit the program

    print("2. Cleaning data...")
    try:
        prepared_data = prepare_tbi_data(df_original)
        print("Data preparation complete.")
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return  # Exit the program

    print("3. Plotting data before cleaning...")
    try:
        plot_before_cleaning(df_original)
        print("Data plot before cleaning complete.")
    except Exception as e:
        print(f"Error plotting data before cleaning: {e}")
        # You can choose whether to exit the program here, depending on the severity of the error
        # return

    print("4. Plotting data after cleaning...")
    try:
        plot_after_cleaning(prepared_data, df_original)
        print("Data plot after cleaning complete.")
    except Exception as e:
        print(f"Error plotting data after cleaning: {e}")
        # You can choose whether to exit the program here, depending on the severity of the error
        # return

    tex_file = "../report/lab1.tex"  
    try:
        tex_to_pdf(tex_file, "../report/lab1-generated")  
        print("PDF created in the 'report/lab1-generated' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Main function completed.")


if __name__ == "__main__":
    main()