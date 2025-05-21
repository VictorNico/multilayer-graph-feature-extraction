# 05_report_generation.py
import pandas as pd
import configparser
from modules.report import generate_report

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    config = load_config("configs/config.ini")

    result_path = config["REPORT"]["result_path"]
    report_path = config["REPORT"]["report_output"]
    email_enabled = config.getboolean("REPORT", "send_email")

    df_results = pd.read_csv(result_path)
    report_file = generate_report(df_results, output_path=report_path)

    print(f"Rapport généré : {report_file}")

    if email_enabled:
        from modules.emailer import send_email
        recipient = config["REPORT"]["recipient"]
        send_email(to=recipient, subject="Rapport MLNA", attachment=report_file)
        print("Rapport envoyé par mail.")

if __name__ == "__main__":
    main()
