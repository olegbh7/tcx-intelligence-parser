import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import os

class TCXParser:
    """
    A professional-grade TCX to CSV parser designed for data scientists and athletes.
    Focuses on high-integrity extraction of heart rate and time-series data.
    """
    def __init__(self, input_dir='input', output_dir='output'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ns = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_file(self, file_path):
        """
        Parses a single TCX file and returns a cleaned DataFrame.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            records = []
            trackpoints = root.findall('.//ns:Trackpoint', self.ns)
            
            for tp in trackpoints:
                time_node = tp.find('ns:Time', self.ns)
                hr_node = tp.find('ns:HeartRateBpm/ns:Value', self.ns)
                
                if time_node is not None and hr_node is not None:
                    records.append({
                        'timestamp': time_node.text,
                        'heart_rate': int(hr_node.text)
                    })
            
            df = pd.DataFrame(records)
            if df.empty:
                return None
                
            # Process Universal Time Formats
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['datetime'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df['unix_epoch'] = df['timestamp'].astype(np.int64) // 10**9
            df['relative_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
            
            return df[['datetime', 'unix_epoch', 'relative_seconds', 'heart_rate']]
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def export_all(self):
        """
        Finds all TCX files in the input directory and exports them as cleaned CSVs.
        """
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.tcx')]
        print(f"--- Starting TCX Batch Parser ({len(files)} files detected) ---")
        
        for file_name in files:
            full_path = os.path.join(self.input_dir, file_name)
            df = self.parse_file(full_path)
            
            if df is not None:
                # Meta-data for filename
                duration_min = int(df['relative_seconds'].iloc[-1] / 60)
                date_str = pd.to_datetime(df['datetime'].iloc[0]).strftime('%Y-%m-%d')
                
                output_name = f"{date_str}_{duration_min}min_parsed.csv"
                df.to_csv(os.path.join(self.output_dir, output_name), index=False)
                print(f"✓ Processed: {file_name} -> {output_name}")
            else:
                print(f"✗ Failed: {file_name} (No valid data)")
        
        print(f"--- Batch Processing Complete. Files stored in '{self.output_dir}' ---")

if __name__ == "__main__":
    # Create input folder if it doesn't exist for the user
    if not os.path.exists('input'):
        os.makedirs('input')
        print("Created 'input' folder. Place your .tcx files there.")
    else:
        parser = TCXParser()
        parser.export_all()
