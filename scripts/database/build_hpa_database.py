"""
build_hpa_database.py

Builds a local HPA (Human Protein Atlas) SQLite database using a TSV and XML data.
Preserves the original extraction logic for antibodies, tissues, images, patients, and reliability.
"""

import os
import csv
import sqlite3
import argparse
import json
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError
from tqdm import tqdm
import time
import tkinter as tk
from tkinter import filedialog

from .. import config_helper


class HPADatabaseBuilder:
    def __init__(self, config):
        self.config_data = config
        self.database_filename = self.config_data['paths']['database_path']

        self.antibody = None
        self.tissue = None
        self.element = None
        self.verification = None
        self.tissueCell_tuple2 = None
        self.find_gene = None  # Optional: filter a specific gene

        self.a_list = []
        self.i_list = []
        self.a_list_dupe = []

        self.version = ""

        self.build_database()

    # ------------------------
    # TSV selection
    # ------------------------
    def select_tsv_file(self):
        tsv_path = self.config_data['paths'].get('tsv_file')
        if not tsv_path or not os.path.exists(tsv_path):
            root = tk.Tk()
            root.withdraw()
            tsv_path = filedialog.askopenfilename(
                title="Select HPA TSV file",
                filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")]
            )
            root.destroy()
        return tsv_path

    # ------------------------
    # Main DB builder
    # ------------------------
    def build_database(self):
        tsv_path = self.select_tsv_file()
        if not tsv_path:
            print("No TSV file selected. Exiting.")
            return

        with open(tsv_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f) - 1

        with open(tsv_path, "r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            next(f)  # skip header

            for line in tqdm(tsv_reader, total=total_lines, desc="Processing genes"):
                gene_name, gene_ID, gene_ab = line[0], line[2], line[15]
                if self.find_gene and gene_name != self.find_gene:
                    continue
                url = f"https://www.proteinatlas.org/{gene_ID}.xml"
                self.process_antibody_data(url, gene_name, gene_ID)

        self.write_to_db()
        print(f"Database build complete: {self.database_filename}")

    # ------------------------
    # Antibody XML processing
    # ------------------------
    def process_antibody_data(self, url, gene_name, gene_ID, ab_filter=""):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = urllib.request.urlopen(url, timeout=100).read().decode("utf-8")
                self.element = ET.fromstring(response)

                for self.antibody in self.element.findall(f".//*antibody{ab_filter}"):
                    antibody_id = self.antibody.attrib.get("id", "")
                    for tissueExpression in self.antibody:
                        if tissueExpression.attrib.get("assayType") == "tissue":
                            self.process_tissue_data(tissueExpression, gene_name, gene_ID, antibody_id)
                return
            except (HTTPError, URLError, TimeoutError) as e:
                print(f"Error fetching {url} (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        print(f"Skipping {url} after {max_retries} failed attempts.")

    def process_tissue_data(self, tissueExpression, gene_name, gene_ID, antibody_id):
        for data in tissueExpression.findall(".//data"):
            for self.tissue in data.findall(".//tissue[@organ='Skin']"):
                if self.tissue.text and "Skin" in self.tissue.text:
                    self.process_image_urls(data, gene_name, gene_ID, antibody_id)

    def process_image_urls(self, data, gene_name, gene_ID, antibody_id):
        # Get verification and tissueCell2
        if self.verification is None and self.tissueCell_tuple2 is None:
            self.verification, self.tissueCell_tuple2 = self.process_verificationANDtissueCell()

        image_elements = data.findall(".//imageUrl")
        if not image_elements:
            # Version fallback
            if not self.version:
                self.version = "v19"
                fallback_url = f"https://{self.version}.proteinatlas.org/{gene_ID}.xml"
                print(f"Falling back to {fallback_url}")
                self.process_antibody_data(fallback_url, gene_name, gene_ID, f"/[@id='{antibody_id}']")
            return

        for imageUrl in image_elements:
            tissueCell_tuples = []
            for tissueCell in data.findall(".//tissueCell"):
                cellType = tissueCell.find('cellType')
                cellType = cellType.text.strip() if cellType is not None else ""
                staining = tissueCell.find('level[@type="staining"]')
                staining = staining.text.strip() if staining is not None else ""
                intensity = tissueCell.find('level[@type="intensity"]')
                intensity = intensity.text.strip() if intensity is not None else ""
                quantity = tissueCell.find('quantity')
                quantity = quantity.text.strip() if quantity is not None else ""
                location = tissueCell.find('location')
                location = location.text.strip() if location is not None else ""
                tissueCell_tuples.append((cellType, staining, intensity, quantity, location))

            sex, age, patient_id, snomed_tuples = self.collect_patient_data(data, imageUrl.text)

            # Build final entry
            entry = [
                gene_name,
                gene_ID,
                antibody_id,
                self.tissue.text,
                imageUrl.text,
                sex,
                age,
                patient_id,
                str(snomed_tuples),
                str(tissueCell_tuples),
                str(self.tissueCell_tuple2),
                self.verification,
            ]
            self.append_unique_entry(entry)

    def process_verificationANDtissueCell(self):
        tissueExpression = self.element.find(
            ".//tissueExpression[@source='HPA'][@technology='IHC'][@assayType='tissue']"
        )
        tissueCell_tuples = ()
        if tissueExpression is not None:
            verification = tissueExpression.find(".//verification[@type='reliability']")
            if verification is None:
                verification = tissueExpression.find(".//verification[@type='validation']")
            verification_txt = verification.text if verification is not None else ""

            for data in tissueExpression.findall(".//data"):
                for tissue in data.findall(".//tissue[@organ='Skin']"):
                    if tissue is not None and self.tissue.text == tissue.text:
                        for tissueCell in data.findall(".//tissueCell"):
                            cellType = tissueCell.find('cellType').text.strip() if tissueCell.find('cellType') is not None else ""
                            expression_level = tissueCell.find('level[@type="expression"]')
                            if expression_level is None:
                                expression = tissueCell.find('level[@type="staining"]').text.strip() if tissueCell.find('level[@type="staining"]') is not None else ""
                            else:
                                expression = expression_level.text.strip()
                            tissueCell_tuples += ((cellType, expression),)
        else:
            verification_txt = ""
        return verification_txt, tissueCell_tuples

    def collect_patient_data(self, data, img_url):
        for patient in data.findall(".//patient"):
            for image in patient.findall(".//imageUrl"):
                if image.text == img_url:
                    sex = patient.find(".//sex").text if patient.find(".//sex") is not None else ""
                    age = patient.find(".//age").text if patient.find(".//age") is not None else ""
                    patient_id = patient.find(".//patientId").text if patient.find(".//patientId") is not None else ""

                    snomed_tuples = ()
                    for snomed in patient.findall(".//snomed"):
                        desc = snomed.get("tissueDescription", "").strip()
                        code = snomed.get("snomedCode", "").strip()
                        snomed_tuples += ((desc, code),)
                    return sex, age, patient_id, snomed_tuples
        return "", "", "", ()

    def append_unique_entry(self, entry):
        tuple_entry = tuple(entry)
        if tuple_entry not in {tuple(e) for e in self.a_list}:
            self.a_list.append(entry)
            self.a_list_dupe = self.a_list[:]

    def write_to_db(self):
        conn = sqlite3.connect(self.database_filename)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS elGenes (
                mainID INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                ID TEXT,
                antibody TEXT,
                tissue TEXT,
                url TEXT,
                sex TEXT,
                age TEXT,
                patientID TEXT,
                snomed_tuple TEXT,
                tissueCell TEXT,
                tissueCell2 TEXT,
                reliability TEXT
            )"""
        )
        if self.a_list_dupe:
            c.executemany(
                "INSERT INTO elGenes (name, ID, antibody, tissue, url, sex, age, patientID, snomed_tuple, tissueCell, tissueCell2, reliability) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self.a_list_dupe,
            )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HPA database from XML/TSV data")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_path = os.path.join(PROJECT_ROOT, "config", "base_config.yaml")
    hpa_path = os.path.join(PROJECT_ROOT, "config", "hpa_database_config.yaml")

    config = config_helper.ConfigLoader.load_config(base_path, hpa_path)
    HPADatabaseBuilder(config)
