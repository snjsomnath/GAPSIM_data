# -----------------------------------------------------
# fetcher.py
# -----------------------------------------------------
# Description: Contains the helper functions used to fetch data from the API.
# Author: Sanjay Somanath
# Last Modified: 2023-10-23
# Version: 0.0.1
# License: MIT License
# Contact: sanjay.somanath@chalmers.se
# Contact: snjsomnath@gmail.com
# -----------------------------------------------------
# Module Metadata:
__name__ = "tripsender.fetcher"
__package__ = "tripsender"
__version__ = "0.0.1"
# -----------------------------------------------------

# Importing libraries
import json
import requests
import logging
from tripsender.logconfig import setup_logging

logger = setup_logging(__name__)

def fetch_housetype_data(year, area):
  """
  Fetches the data for the number of households in different house types in a given area for a given year.
  """
    url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Befolkning/Hushåll/31_HHStorlHustyp_PRI.px"
    query = {
  "query": [
    {
      "code": "Område",
      "selection": {
        "filter": "item",
        "values": [
          area
        ]
      }
    },
    {
      "code": "Hushållsstorlek",
      "selection": {
        "filter": "item",
        "values": [
          "1 person",
          "2 personer",
          "3 personer",
          "4 personer",
          "5 personer",
          "6 eller fler personer",
          "Uppgift saknas"
        ]
      }
    },
    {
      "code": "Hustyp",
      "selection": {
        "filter": "item",
        "values": [
          "Småhus",
          "Flerbostadshus",
          "Specialbostad, övriga hus",
          "Uppgift saknas"
        ]
      }
    },
    {
      "code": "År",
      "selection": {
        "filter": "item",
        "values": [
          str(year)
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
    }

    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")

    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_population_data(year, area):
  """
  Fetches the data for the population in a given area for a given year.
  """
    url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Befolkning/Folkmängd/Folkmängd helår/60_FolkmHHStallning_PRI.px"

    query = {
        "query": [
            {
                "code": "Område",
                "selection": {
                    "filter": "item",
                    "values": [
                        area
                    ]
                }
            },
            {
                "code": "Kön",
                "selection": {
                    "filter": "item",
                    "values": [
                        "Män",
                        "Kvinnor"
                    ]
                }
            },
            {
                "code": "År",
                "selection": {
                    "filter": "item",
                    "values": [
                        str(year)
                    ]
                }
            }
        ],
        "response": {
            "format": "json"
        }
    }

    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")

    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_younger_children_data(year,area):
  """
  Fetches the data for the number of households with children under 18 years in a given area for a given year.
  """
    url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Befolkning/Hushåll/10_HHTypBarnU18_PRI.px"
    query = {
    "query": [
    {
      "code": "Område",
      "selection": {
        "filter": "item",
        "values": [
          area
        ]
      }
    },
    {
      "code": "Hushållstyp",
      "selection": {
        "filter": "item",
        "values": [
          "Ensamstående",
          "Sammanboende",
          "Övriga hushåll",
          "Uppgift saknas"
        ]
      }
    },
    {
      "code": "År",
      "selection": {
        "filter": "item",
        "values": [
          str(year)
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
  }
    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")

    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_older_children_data(year,area):
  """
  Fetches the data for the number of households with children over 18 years in a given area for a given year.
  """
    url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Befolkning/Hushåll/20_Hushallstyp_PRI.px"
    query = {
    "query": [
        {
        "code": "Område",
        "selection": {
            "filter": "item",
            "values": [
            area
            ]
        }
        },
        {
        "code": "År",
        "selection": {
            "filter": "item",
            "values": [
            year
            ]
        }
        }
    ],
    "response": {
        "format": "json"
    }
    }
    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")

    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_car_data(year,area):
    url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Övrigt/Personbilar/10_Bilar_PRI.px"

    query = {
  "query": [
    {
      "code": "Område",
      "selection": {
        "filter": "item",
        "values": [
          area
        ]
      }
    },
    {
      "code": "Tabellvärde",
      "selection": {
        "filter": "item",
        "values": [
          "Personbilar"
        ]
      }
    },
    {
      "code": "År",
      "selection": {
        "filter": "item",
        "values": [
          str(year)
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
    }

    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")
    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_municipal_children_data(year):
    url = "https://api.scb.se/OV0104/v1/doris/en/ssd/START/BE/BE0101/BE0101S/HushallT05"

    query = {
    "query": [
        {
        "code": "Region",
        "selection": {
            "filter": "vs:RegionKommun07",
            "values": [
            "1480"
            ]
        }
        },
        {
        "code": "Hushallstyp",
        "selection": {
            "filter": "item",
            "values": [
            "ESUB",
            "ESMB25",
            "ESMB24",
            "SMUB",
            "SBMB25",
            "SBMB24",
            "OVRIUB",
            "ÖMB25",
            "ÖMB24",
            "SAKNAS"
            ]
        }
        },
        {
        "code": "Barn",
        "selection": {
            "filter": "item",
            "values": [
            "UB",
            "M1B",
            "M2B",
            "M3+B",
            "SAKNAS"
            ]
        }
        },
        {
        "code": "ContentsCode",
        "selection": {
            "filter": "item",
            "values": [
            "BE0101C$"
            ]
        }
        },
        {
        "code": "Tid",
        "selection": {
            "filter": "item",
            "values": [
            year
            ]
        }
        }
    ],
    "response": {
        "format": "json"
    }
    }

    response = requests.post(url, json=query)
    response_code = response.status_code
    data = response.content.decode("utf-8-sig")
    #print("Response Code:", response_code)
    return json.loads(data)

def fetch_total_households(year,area):
  url = "http://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Befolkning/Hushåll/31_HHStorlHustyp_PRI.px"
  query = {
            "query": [
              {
                "code": "Område",
                "selection": {
                  "filter": "item",
                  "values": [
                    area
                  ]
                }
              },
              {
                "code": "År",
                "selection": {
                  "filter": "item",
                  "values": [
                    int(year)
                  ]
                }
              }
            ],
            "response": {
              "format": "json"
            }
          }
    
  response = requests.post(url, json=query)
  response_code = response.status_code
  data = response.content.decode("utf-8-sig")
  #print("Response Code:", response_code)
  d = json.loads(data)
  return int(d['data'][0]['values'][0])

def fetch_primary_status(year,area):
  url = "https://pxweb.goteborg.se/api/v1/sv/1. Göteborg och dess delområden/Primärområden/Inkomst och utbildning/Inkomster/Förvärvsinkomster etc/20_HuvudInk_PRI.px"
  query = {
  "query": [
    {
      "code": "Område",
      "selection": {
        "filter": "item",
        "values": [
          area
        ]
      }
    },
    {
      "code": "År",
      "selection": {
        "filter": "item",
        "values": [
          year
        ]
      }
    }
  ],
  "response": {
    "format": "json"
  }
}
  #print(query)
  response = requests.post(url, json=query)
  response_code = response.status_code
  raw_data = response.content.decode("utf-8-sig")
  #print("Response Code:", response_code)
  data = json.loads(raw_data)
  d = {}
  for i in range(len(data["data"])):
      working_key = "Förvärvsarbete"
      studying_key = "Studerande"
      work = "WORK"
      study = "STUDY"
      other = "INACTIVE"
      k = data["data"][i]["key"][1]
      v = data["data"][i]["values"][0]
      if k == working_key:
          d[work] = v
      elif k == studying_key:
          d[study] = v
      else:
          d[other] = v
  return d