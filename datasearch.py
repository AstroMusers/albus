import csv

file_path = 'data_outputs/noninjected_transits_output4.csv'
with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    max_snr_row = None
    max_snr = float('-inf')

    for row in reader:
        try:
            snr = float(row["snr"])
            if snr > max_snr:
                max_snr = snr
                max_snr_row = row
        except ValueError:
            continue  # Skip rows where SNR is not a valid number

print(max_snr_row)