"""
This module pulls and saves data on fundamentals from CRSP and Compustat.
It pulls fundamentals data from Compustat needed to calculate
book equity, and the data needed from CRSP to calculate market equity.

Note: This code uses the new CRSP CIZ format. Information
about the differences between the SIZ and CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

For information about Compustat variables, see:
https://wrds-www.wharton.upenn.edu/documents/1583/Compustat_Data_Guide.pdf

For more information about variables in CRSP, see:
https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
I don't think this is updated for the new CIZ format, though.

Here is some information about the old SIZ CRSP format:
https://wrds-www.wharton.upenn.edu/documents/1095/CRSP_Flat_File_formats_and_notes.pdf


The following is an outdated programmer's guide to CRSP:
https://wrds-www.wharton.upenn.edu/documents/400/CRSP_Programmers_Guide.pdf


"""
from pathlib import Path

import pandas as pd
import wrds

from settings import config

# ==============================================================================================
# Global Configuration
# ==============================================================================================

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

# ==============================================================================================
# Compustat Data Functions
# ==============================================================================================

description_compustat = {
    "gvkey": "Global Company Key",
    "datadate": "Data Date",
    "at": "Assets - Total",
    "sale": "Sales/Revenue",
    "cogs": "Cost of Goods Sold",
    "xsga": "Selling, General and Administrative Expense",
    "xint": "Interest Expense, Net",
    "pstkl": "Preferred Stock - Liquidating Value",
    "txditc": "Deferred Taxes and Investment Tax Credit",
    "pstkrv": "Preferred Stock - Redemption Value",
    # This item represents the total dollar value of the net number of
    # preferred shares outstanding multiplied by the voluntary
    # liquidation or redemption value per share.
    "seq": "Stockholders' Equity - Parent",
    "pstk": "Preferred/Preference Stock (Capital) - Total",
    "indfmt": "Industry Format",
    "datafmt": "Data Format",
    "popsrc": "Population Source",
    "consol": "Consolidation",
}


def pull_Compustat(wrds_username=WRDS_USERNAME, vars_str=None):
    """
    See description_Compustat for a description of the variables.
    Annual Compustat fundamental data.
    """
    if vars_str is not None:
        vars_str = ", ".join(vars_str)
    else: 
        vars_str = "gvkey, datadate, at, sale, cogs, xsga, xint, pstkl, txditc, pstkrv, seq, pstk, ni, sich, dp, ebit"

    sql_query = """
        SELECT 
            {vars_str}
        FROM 
            comp.funda
        WHERE 
            indfmt='INDL' AND -- industrial reporting format (not financial services format)
            datafmt='STD' AND -- only standardized records
            popsrc='D' AND -- only from domestic sources
            consol='C' AND -- consolidated financial statements
            datadate >= '01/01/1959'
        """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     comp = db.raw_sql(sql_query, date_cols=["datadate"])
    db = wrds.Connection(wrds_username=wrds_username)
    comp = db.raw_sql(sql_query, date_cols=["datadate"])
    db.close()

    comp["year"] = comp["datadate"].dt.year
    return comp


description_crsp = {
    "permno": "Permanent Number - A unique identifier assigned by CRSP to each security.",
    "permco": "Permanent Company - A unique company identifier assigned by CRSP that remains constant over time for a given company.",
    "mthcaldt": "Calendar Date - The date for the monthly data observation.",
    "issuertype": "Issuer Type - Classification of the issuer, such as corporate or government.",
    "securitytype": "Security Type - General classification of the security, e.g., stock or bond.",
    "securitysubtype": "Security Subtype - More specific classification of the security within its type.",
    "sharetype": "Share Type - Classification of the equity share type, e.g., common stock, preferred stock.",
    "usincflg": "U.S. Incorporation Flag - Indicator of whether the company is incorporated in the U.S.",
    "primaryexch": "Primary Exchange - The primary stock exchange where the security is listed.",
    "conditionaltype": "Conditional Type - Indicator of any conditional issues related to the security.",
    "tradingstatusflg": "Trading Status Flag - Indicator of the trading status of the security, e.g., active, suspended.",
    "mthret": "Monthly Return - The total return of the security for the month, including dividends.",
    "mthretx": "Monthly Return Excluding Dividends - The return of the security for the month, excluding dividends.",
    "shrout": "Shares Outstanding - The number of outstanding shares of the security.",
    "mthprc": "Monthly Price - The price of the security at the end of the month.",
}


description_CRSP_Comp_link = {
    "gvkey": "Global Company Key - A unique identifier for companies in the Compustat database.",
    "permno": "Permanent Number - A unique stock identifier assigned by CRSP to each security.",
    "linktype": "Link Type - Indicates the type of linkage between CRSP and Compustat records. 'L' types refer to links considered official by CRSP.",
    "linkprim": "Primary Link Indicator - Specifies whether the link is a primary identified by Compustat ('P'), primary assigned by CRSP ('C') connection between the databases, or secondary ('J') for secondary securities for each company (used for total market cap). Primary links are direct matches between CRSP and Compustat entities, while secondary links may represent subsidiary relationships or other less direct connections.",
    "linkdt": "Link Date Start - The starting date for which the linkage between CRSP and Compustat data is considered valid.",
    "linkenddt": "Link Date End - The ending date for which the linkage is considered valid. A blank or high value (e.g., '2099-12-31') indicates that the link is still valid as of the last update.",
}


def pull_CRSP_Comp_link_table(wrds_username=WRDS_USERNAME):
    """ 
    Pull the CRSP-Compustat link table.
    https://wrds-www.wharton.upenn.edu/pages/wrds-research/database-linking-matrix/linking-crsp-with-compustat/
    """
    sql_query = """
        SELECT 
            gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM 
            crsp.ccmxpf_linktable
        WHERE 
            substr(linktype,1,1)='L' AND 
            (linkprim ='C' OR linkprim='P')
        """
    db = wrds.Connection(wrds_username=wrds_username)
    ccm = db.raw_sql(sql_query, date_cols=["linkdt", "linkenddt"])
    db.close()
    return ccm


def load_Compustat(data_dir=RAW_DATA_DIR):
    path = Path(data_dir) / "Compustat.parquet"
    comp = pd.read_parquet(path)
    return comp


def load_CRSP_Comp_Link_Table(data_dir=RAW_DATA_DIR):
    path = Path(data_dir) / "CRSP_Comp_Link_Table.parquet"
    ccm = pd.read_parquet(path)
    return ccm


def _demo():
    comp = load_Compustat(data_dir=RAW_DATA_DIR)
    ccm = load_CRSP_Comp_Link_Table(data_dir=RAW_DATA_DIR)


if __name__ == "__main__":
    comp = pull_Compustat(wrds_username=WRDS_USERNAME)
    comp.to_parquet(RAW_DATA_DIR / "Compustat.parquet")

    ccm = pull_CRSP_Comp_link_table(wrds_username=WRDS_USERNAME)
    ccm.to_parquet(RAW_DATA_DIR / "CRSP_Comp_Link_Table.parquet")