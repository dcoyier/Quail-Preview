# State of the Union Dataset Codebook and CAP Topic Codes

This standalone reference is for the State of the Union dataset file `US-Exec_SOTU_2025.csv`.

## Dataset Scope

The State of the Union dataset is organized at the quasi-statement level. A quasi-statement is a short statement unit split at sentence punctuation or semicolons, so a single speech can produce many rows. Each row contains the text of one quasi-statement plus metadata about the address and policy-content topic codes.

Each policy-content entry is assigned one content code. Non-policy statements are marked separately through `filter_PolicySentence` and special topic-code values.

## Column Naming Note

Some releases of this dataset use older variable names such as `KeyID`, `Count`, `PolicySentence`, `USMajorTopic`, `USSubTopicCode`, `MajorTopic`, and `SubTopicCode`. The 2025 CSV uses cleaned snake_case names. The mapping is shown below.

In the 2025 CSV, `pap_majortopic` always matches `majortopic`, and `pap_subtopic` always matches `subtopic` for all non-empty rows. In this file they are duplicate values, but conceptually the `pap_*` columns refer to Policy Agendas Project topic coding and the non-prefixed columns refer to the Comparative Agendas Project harmonized topic coding.

## Dataset Variables

| CSV column | Alternate name | Meaning / coding notes |
|---|---|---|
| `id` | `KeyID` | Unique row identifier. Not substantively meaningful. |
| `doc_count` | `Count` | Quasi-statement number within that year's address. |
| `filter_PolicySentence` | `PolicySentence` | `1` if the quasi-statement has policy content; `0` otherwise. |
| `date` | `Date` | Date the address was delivered. |
| `oral_delivery` | `OralDelivery` | `1` if delivered orally; `0` if submitted in writing. |
| `outgoing` | `Outgoing` | `1` if the president was outgoing and giving a final address before transition; `0` otherwise. |
| `congress` | `Congress` | Congress term in which the address was delivered. |
| `president` | `President` | President who delivered the address. |
| `pres_party` | `PresParty` | President's party. `100` means Democrat and `200` means Republican. |
| `divided` | `Divided` | `1` for divided party control of legislative and executive branches; `0` for unified control. |
| `control_house` | `ControlHouse` | Party controlling the House. `100` means Democratic control and `200` means Republican control. |
| `control_senate` | `ControlSenate` | Party controlling the Senate. `100` means Democratic control and `200` means Republican control. |
| `year` | `Year` | Address year. |
| `month` | `Month` | Address month. |
| `day` | `Day` | Address day. |
| `source` | `Source` | Source used by the Policy Agendas Project for that year's address. |
| `description` | `Description` | Text of the quasi-statement. |
| `pap_majortopic` | `USMajorTopic` | Policy Agendas Project major topic code for the quasi-statement. |
| `pap_subtopic` | `USSubTopicCode` | Policy Agendas Project subtopic code for the quasi-statement. |
| `majortopic` | `MajorTopic` | Comparative Agendas Project major topic code corresponding to the PAP topic code. |
| `subtopic` | `SubTopicCode` | Comparative Agendas Project subtopic code corresponding to the PAP topic code. |

## Topic-Code Fields

Use `description` as the text being coded. Use the topic columns as categorical labels.

| Field | Level | Project context | How to interpret |
|---|---|---|---|
| `pap_majortopic` | Major topic | Policy Agendas Project | Broad U.S. Policy Agendas topic area. |
| `pap_subtopic` | Subtopic | Policy Agendas Project | More specific issue category nested under the major topic. |
| `majortopic` | Major topic | Comparative Agendas Project | CAP harmonized major topic corresponding to the PAP code. |
| `subtopic` | Subtopic | Comparative Agendas Project | CAP harmonized subtopic corresponding to the PAP code. |

Special values:

| Value | Where | Meaning |
|---|---|---|
| `-555` | Topic/subtopic fields | Non-policy or no substantive topic code. The 2025 CSV uses `-555` in subtopic fields and also in major-topic fields for many non-policy rows. |
| `0` | Major-topic fields in the 2025 CSV | Non-policy / no substantive major topic for some recent rows. These rows also have `filter_PolicySentence = 0` and `subtopic = -555`. |
| blank row | CSV artifact | The file contains one fully blank trailing row. Ignore it. |

## Major Topic Codes

| Major code | Major topic |
|---:|---|
| 1 | Macroeconomics |
| 2 | Civil Rights |
| 3 | Health |
| 4 | Agriculture |
| 5 | Labor |
| 6 | Education |
| 7 | Environment |
| 8 | Energy |
| 9 | Immigration |
| 10 | Transportation |
| 12 | Law and Crime |
| 13 | Social Welfare |
| 14 | Housing |
| 15 | Domestic Commerce |
| 16 | Defense |
| 17 | Technology |
| 18 | Foreign Trade |
| 19 | International Affairs |
| 20 | Government Operations |
| 21 | Public Lands |
| 23 | Culture |

## CAP Subtopic Codes

This table gives the CAP topic and subtopic code labels used to interpret `majortopic`, `subtopic`, `pap_majortopic`, and `pap_subtopic`.

| Major code | Major topic | Subtopic code | Subtopic label |
|---:|---|---:|---|
| 1 | Macroeconomics | 100 | General |
| 1 | Macroeconomics | 101 | Interest Rates |
| 1 | Macroeconomics | 103 | Unemployment Rate |
| 1 | Macroeconomics | 104 | Monetary Policy |
| 1 | Macroeconomics | 105 | National Budget |
| 1 | Macroeconomics | 107 | Tax Code |
| 1 | Macroeconomics | 108 | Industrial Policy |
| 1 | Macroeconomics | 110 | Price Control |
| 1 | Macroeconomics | 199 | Other |
| 2 | Civil Rights | 200 | General |
| 2 | Civil Rights | 201 | Minority Discrimination |
| 2 | Civil Rights | 202 | Gender Discrimination |
| 2 | Civil Rights | 204 | Age Discrimination |
| 2 | Civil Rights | 205 | Handicap Discrimination |
| 2 | Civil Rights | 206 | Voting Rights |
| 2 | Civil Rights | 207 | Freedom of Speech |
| 2 | Civil Rights | 208 | Right to Privacy |
| 2 | Civil Rights | 209 | Anti-Government |
| 2 | Civil Rights | 299 | Other |
| 3 | Health | 300 | General |
| 3 | Health | 301 | Health Care Reform |
| 3 | Health | 302 | Insurance |
| 3 | Health | 321 | Drug Industry |
| 3 | Health | 322 | Medical Facilities |
| 3 | Health | 323 | Insurance Providers |
| 3 | Health | 324 | Medical Liability |
| 3 | Health | 325 | Manpower |
| 3 | Health | 331 | Disease Prevention |
| 3 | Health | 332 | Infants and Children |
| 3 | Health | 333 | Mental Health |
| 3 | Health | 334 | Long-term Care |
| 3 | Health | 335 | Drug Coverage and Cost |
| 3 | Health | 341 | Tobacco Abuse |
| 3 | Health | 342 | Drug and Alcohol Abuse |
| 3 | Health | 398 | R&D |
| 3 | Health | 399 | Other |
| 4 | Agriculture | 400 | General |
| 4 | Agriculture | 401 | Trade |
| 4 | Agriculture | 402 | Subsidies to Farmers |
| 4 | Agriculture | 403 | Food Inspection & Safety |
| 4 | Agriculture | 404 | Food Marketing & Promotion |
| 4 | Agriculture | 405 | Animal and Crop Disease |
| 4 | Agriculture | 408 | Fisheries & Fishing |
| 4 | Agriculture | 498 | R&D |
| 4 | Agriculture | 499 | Other |
| 5 | Labor | 500 | General |
| 5 | Labor | 501 | Worker Safety |
| 5 | Labor | 502 | Employment Training |
| 5 | Labor | 503 | Employee Benefits |
| 5 | Labor | 504 | Labor Unions |
| 5 | Labor | 505 | Fair Labor Standards |
| 5 | Labor | 506 | Youth Employment |
| 5 | Labor | 529 | Migrant and Seasonal |
| 5 | Labor | 599 | Other |
| 6 | Education | 600 | General |
| 6 | Education | 601 | Higher |
| 6 | Education | 602 | Elementary & Secondary |
| 6 | Education | 603 | Underprivileged |
| 6 | Education | 604 | Vocational |
| 6 | Education | 606 | Special |
| 6 | Education | 607 | Excellence |
| 6 | Education | 698 | R&D |
| 6 | Education | 699 | Other |
| 7 | Environment | 700 | General |
| 7 | Environment | 701 | Drinking Water |
| 7 | Environment | 703 | Waste Disposal |
| 7 | Environment | 704 | Hazardous Waste |
| 7 | Environment | 705 | Air Pollution |
| 7 | Environment | 707 | Recycling |
| 7 | Environment | 708 | Indoor Hazards |
| 7 | Environment | 709 | Species & Forest |
| 7 | Environment | 711 | Land and Water Conservation |
| 7 | Environment | 798 | R&D |
| 7 | Environment | 799 | Other |
| 8 | Energy | 800 | General |
| 8 | Energy | 801 | Nuclear |
| 8 | Energy | 802 | Electricity |
| 8 | Energy | 803 | Natural Gas & Oil |
| 8 | Energy | 805 | Coal |
| 8 | Energy | 806 | Alternative & Renewable |
| 8 | Energy | 807 | Conservation |
| 8 | Energy | 898 | R&D |
| 8 | Energy | 899 | Other |
| 9 | Immigration | 900 | Immigration |
| 10 | Transportation | 1000 | General |
| 10 | Transportation | 1001 | Mass |
| 10 | Transportation | 1002 | Highways |
| 10 | Transportation | 1003 | Air Travel |
| 10 | Transportation | 1005 | Railroad Travel |
| 10 | Transportation | 1007 | Maritime |
| 10 | Transportation | 1010 | Infrastructure |
| 10 | Transportation | 1098 | R&D |
| 10 | Transportation | 1099 | Other |
| 12 | Law and Crime | 1200 | General |
| 12 | Law and Crime | 1201 | Agencies |
| 12 | Law and Crime | 1202 | White Collar Crime |
| 12 | Law and Crime | 1203 | Illegal Drugs |
| 12 | Law and Crime | 1204 | Court Administration |
| 12 | Law and Crime | 1205 | Prisons |
| 12 | Law and Crime | 1206 | Juvenile Crime |
| 12 | Law and Crime | 1207 | Child Abuse |
| 12 | Law and Crime | 1208 | Family Issues |
| 12 | Law and Crime | 1210 | Criminal & Civil Code |
| 12 | Law and Crime | 1211 | Crime Control |
| 12 | Law and Crime | 1227 | Police |
| 12 | Law and Crime | 1299 | Other |
| 13 | Social Welfare | 1300 | General |
| 13 | Social Welfare | 1302 | Low-Income Assistance |
| 13 | Social Welfare | 1303 | Elderly Assistance |
| 13 | Social Welfare | 1304 | Disabled Assistance |
| 13 | Social Welfare | 1305 | Volunteer Associations |
| 13 | Social Welfare | 1308 | Child Care |
| 13 | Social Welfare | 1399 | Other |
| 14 | Housing | 1400 | General |
| 14 | Housing | 1401 | Community Development |
| 14 | Housing | 1403 | Urban Development |
| 14 | Housing | 1404 | Rural Housing |
| 14 | Housing | 1405 | Rural Development |
| 14 | Housing | 1406 | Low-Income Assistance |
| 14 | Housing | 1407 | Veterans |
| 14 | Housing | 1408 | Elderly |
| 14 | Housing | 1409 | Homeless |
| 14 | Housing | 1498 | R&D |
| 14 | Housing | 1499 | Other |
| 15 | Domestic Commerce | 1500 | General |
| 15 | Domestic Commerce | 1501 | Banking |
| 15 | Domestic Commerce | 1502 | Securities & Commodities |
| 15 | Domestic Commerce | 1504 | Consumer Finance |
| 15 | Domestic Commerce | 1505 | Insurance Regulation |
| 15 | Domestic Commerce | 1507 | Bankruptcy |
| 15 | Domestic Commerce | 1520 | Corporate Management |
| 15 | Domestic Commerce | 1521 | Small Businesses |
| 15 | Domestic Commerce | 1522 | Copyrights and Patents |
| 15 | Domestic Commerce | 1523 | Disaster Relief |
| 15 | Domestic Commerce | 1524 | Tourism |
| 15 | Domestic Commerce | 1525 | Consumer Safety |
| 15 | Domestic Commerce | 1526 | Sports Regulation |
| 15 | Domestic Commerce | 1598 | R&D |
| 15 | Domestic Commerce | 1599 | Other |
| 16 | Defense | 1600 | General |
| 16 | Defense | 1602 | Alliances |
| 16 | Defense | 1603 | Intelligence |
| 16 | Defense | 1604 | Readiness |
| 16 | Defense | 1605 | Nuclear Arms |
| 16 | Defense | 1606 | Military Aid |
| 16 | Defense | 1608 | Personnel Issues |
| 16 | Defense | 1610 | Procurement |
| 16 | Defense | 1611 | Installations & Land |
| 16 | Defense | 1612 | Reserve Forces |
| 16 | Defense | 1614 | Hazardous Waste |
| 16 | Defense | 1615 | Civil |
| 16 | Defense | 1616 | Civilian Personnel |
| 16 | Defense | 1617 | Contractors |
| 16 | Defense | 1619 | Foreign Operations |
| 16 | Defense | 1620 | Claims against Military |
| 16 | Defense | 1698 | R&D |
| 16 | Defense | 1699 | Other |
| 17 | Technology | 1700 | General |
| 17 | Technology | 1701 | Space |
| 17 | Technology | 1704 | Commercial Use of Space |
| 17 | Technology | 1705 | Science Transfer |
| 17 | Technology | 1706 | Telecommunications |
| 17 | Technology | 1707 | Broadcast |
| 17 | Technology | 1708 | Weather Forecasting |
| 17 | Technology | 1709 | Computers |
| 17 | Technology | 1798 | R&D |
| 17 | Technology | 1799 | Other |
| 18 | Foreign Trade | 1800 | General |
| 18 | Foreign Trade | 1802 | Trade Agreements |
| 18 | Foreign Trade | 1803 | Exports |
| 18 | Foreign Trade | 1804 | Private Investments |
| 18 | Foreign Trade | 1806 | Competitiveness |
| 18 | Foreign Trade | 1807 | Tariff & Imports |
| 18 | Foreign Trade | 1808 | Exchange Rates |
| 18 | Foreign Trade | 1899 | Other |
| 19 | International Affairs | 1900 | General |
| 19 | International Affairs | 1901 | Foreign Aid |
| 19 | International Affairs | 1902 | Resources Exploitation |
| 19 | International Affairs | 1905 | Developing Countries |
| 19 | International Affairs | 1906 | International Finance |
| 19 | International Affairs | 1910 | Western Europe |
| 19 | International Affairs | 1921 | Specific Country |
| 19 | International Affairs | 1925 | Human Rights |
| 19 | International Affairs | 1926 | Organizations |
| 19 | International Affairs | 1927 | Terrorism |
| 19 | International Affairs | 1929 | Diplomats |
| 19 | International Affairs | 1999 | Other |
| 20 | Government Operations | 2000 | General |
| 20 | Government Operations | 2001 | Intergovernmental Relations |
| 20 | Government Operations | 2002 | Bureaucracy |
| 20 | Government Operations | 2003 | Postal Service |
| 20 | Government Operations | 2004 | Employees |
| 20 | Government Operations | 2005 | Appointments |
| 20 | Government Operations | 2006 | Currency |
| 20 | Government Operations | 2007 | Procurement & Contractors |
| 20 | Government Operations | 2008 | Property Management |
| 20 | Government Operations | 2009 | Tax Administration |
| 20 | Government Operations | 2010 | Scandals |
| 20 | Government Operations | 2011 | Branch Relations |
| 20 | Government Operations | 2012 | Political Campaigns |
| 20 | Government Operations | 2013 | Census & Statistics |
| 20 | Government Operations | 2014 | Capital City |
| 20 | Government Operations | 2015 | Claims against the government |
| 20 | Government Operations | 2030 | National Holidays |
| 20 | Government Operations | 2099 | Other |
| 21 | Public Lands | 2100 | General |
| 21 | Public Lands | 2101 | National Parks |
| 21 | Public Lands | 2102 | Indigenous Affairs |
| 21 | Public Lands | 2103 | Public Lands |
| 21 | Public Lands | 2104 | Water Resources |
| 21 | Public Lands | 2105 | Dependencies & Territories |
| 21 | Public Lands | 2199 | Other |
| 23 | Culture | 2300 | General |

## Practical Use In Quail / Analysis

Recommended field treatment:

| Column | Suggested type | Notes |
|---|---|---|
| `description` | string | Main text field to embed and search. |
| `president`, `source` | string | Useful metadata filters or grouping fields. |
| `year`, `month`, `day`, `congress` | integer | Useful for time slicing and aggregation. |
| `filter_PolicySentence`, `oral_delivery`, `outgoing`, `divided` | integer or boolean-like category | Treat as categorical indicators if comparing groups. |
| `pap_majortopic`, `pap_subtopic`, `majortopic`, `subtopic` | categorical code | Keep as labels/codes, not continuous numeric quantities. |

For most analysis of this specific CSV, use `majortopic` and `subtopic` or use `pap_majortopic` and `pap_subtopic`; they currently contain identical values. Keep both only if you want to preserve the source schema.
