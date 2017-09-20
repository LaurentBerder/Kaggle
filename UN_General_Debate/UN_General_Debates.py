import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from string import punctuation


#Read data
debates = pd.read_csv("C:/Users/galdo/source/repos/Kaggle/UN_General_Debate/un-general-debates.zip", compression='zip')

####### Description ##########
#Have a first look at the data.

debates.head()
debates.describe(include="all")
#We have 7507 speaches, ranging from 1970 to 2015.
#They were spoken by representatives of 199 countries.

debates[["year", "country"]].groupby("year").count().plot(kind="bar")
#It seems like more an more countries are speakers at the General Debates each year, from 1970 (70 countries) to at least 2006 (193 countries).
#This is explained, at first by the end of series of decolonization in the 70s and 80s (more countries break of from their empires to become independent countries), then by the break-up of the Soviet Union after 1989.


####### Text Preparation ##########
#Some special characters are not recognized (wrong encoding), which would prevent further analysis, so let's remove them.
debates['text'] = debates['text'].str.decode(encoding='ascii',errors='ignore').str.lower()

#Now let's transform the text to lists
debates['token'] = debates['text'].apply(word_tokenize)

#I want to find out which countries speakers mention, and will show trends in time, as well as  look at whether certain countries speak more about specific other countries. List found [here](http://www.nationsonline.org/oneworld/country_code_list.htm), and tweaked here and there for better results.
countries = dict((k, v.lower()) for k,v in {
    'AFG': 'Afghanistan', 
    'ALA': 'Aland Islands', 
    'ALB': 'Albania', 
    'DZA': 'Algeria', 
    'ASM': 'American Samoa', 
    'AND': 'Andorra', 
    'AGO': 'Angola', 
    'AIA': 'Anguilla', 
    'ATA': 'Antarctica', 
    'ATG': 'Antigua and Barbuda', 
    'ARG': 'Argentina', 
    'ARM': 'Armenia', 
    'ABW': 'Aruba', 
    'AUS': 'Australia', 
    'AUT': 'Austria', 
    'AZE': 'Azerbaijan', 
    'BHS': 'Bahamas', 
    'BHR': 'Bahrain', 
    'BGD': 'Bangladesh', 
    'BRB': 'Barbados', 
    'BLR': 'Belarus', 
    'BEL': 'Belgium', 
    'BLZ': 'Belize', 
    'BEN': 'Benin', 
    'BMU': 'Bermuda', 
    'BTN': 'Bhutan', 
    'BOL': 'Bolivia', 
    'BIH': 'Bosnia and Herzegovina', 
    'BWA': 'Botswana', 
    'BVT': 'Bouvet Island', 
    'BRA': 'Brazil', 
    'VGB': 'Virgin Islands', 
    'IOT': 'British Indian Ocean Territory', 
    'BRN': 'Brunei', 
    'BGR': 'Bulgaria', 
    'BFA': 'Burkina Faso', 
    'BDI': 'Burundi', 
    'KHM': 'Cambodia', 
    'CMR': 'Cameroon', 
    'CAN': 'Canada', 
    'CPV': 'Cape Verde', 
    'CYM': 'Cayman Islands', 
    'CAF': 'Central Africa', 
    'TCD': 'Chad', 
    'CHL': 'Chile', 
    'CHN': 'China', 
    'HKG': 'Hong Kong', 
    'MAC': 'Macao', 
    'CXR': 'Christmas Island', 
    'CCK': 'Cocos Islands', 
    'COL': 'Colombia', 
    'COM': 'Comoros', 
    'COG': 'Congo', 
    'COD': 'Democratic Republic of Congo', 
    'COK': 'Cook Islands', 
    'CRI': 'Costa Rica', 
    'CIV': "Cote d'Ivoire", 
    'HRV': 'Croatia', 
    'CUB': 'Cuba', 
    'CYP': 'Cyprus', 
    'CZE': 'Czech Republic', 
    'DNK': 'Denmark', 
    'DJI': 'Djibouti', 
    'DMA': 'Dominica', 
    'DOM': 'Dominican Republic', 
    'ECU': 'Ecuador', 
    'EGY': 'Egypt', 
    'SLV': 'El Salvador', 
    'GNQ': 'Equatorial Guinea', 
    'ERI': 'Eritrea', 
    'EST': 'Estonia', 
    'ETH': 'Ethiopia', 
    'FLK': 'Falkland', 
    'FRO': 'Faroe', 
    'FJI': 'Fiji', 
    'FIN': 'Finland', 
    'FRA': 'France', 
    'GUF': 'French Guiana', 
    'PYF': 'French Polynesia', 
    'ATF': 'French Southern Territories', 
    'GAB': 'Gabon', 
    'GMB': 'Gambia', 
    'GEO': 'Georgia', 
    'DEU': 'Germany', 
    'GHA': 'Ghana', 
    'GIB': 'Gibraltar', 
    'GRC': 'Greece', 
    'GRL': 'Greenland', 
    'GRD': 'Grenada', 
    'GLP': 'Guadeloupe', 
    'GUM': 'Guam', 
    'GTM': 'Guatemala', 
    'GGY': 'Guernsey', 
    'GIN': 'Guinea', 
    'GNB': 'Guinea-Bissau', 
    'GUY': 'Guyana', 
    'HTI': 'Haiti', 
    'HMD': 'Heard and Mcdonald Islands', 
    'VAT': 'Vatican', 
    'HND': 'Honduras', 
    'HUN': 'Hungary', 
    'ISL': 'Iceland', 
    'IND': 'India', 
    'IDN': 'Indonesia', 
    'IRN': 'Iran', 
    'IRQ': 'Iraq', 
    'IRL': 'Ireland', 
    'IMN': 'Isle of Man', 
    'ISR': 'Israel', 
    'ITA': 'Italy', 
    'JAM': 'Jamaica', 
    'JPN': 'Japan', 
    'JEY': 'Jersey', 
    'JOR': 'Jordan', 
    'KAZ': 'Kazakhstan', 
    'KEN': 'Kenya', 
    'KIR': 'Kiribati', 
    'PRK': 'North Korea', 
    'KOR': 'South Korea', 
    'KWT': 'Kuwait', 
    'KGZ': 'Kyrgyzstan', 
    'LAO': 'Lao', 
    'LVA': 'Latvia', 
    'LBN': 'Lebanon', 
    'LSO': 'Lesotho', 
    'LBR': 'Liberia', 
    'LBY': 'Libya', 
    'LIE': 'Liechtenstein', 
    'LTU': 'Lithuania', 
    'LUX': 'Luxembourg', 
    'MKD': 'Macedonia', 
    'MDG': 'Madagascar', 
    'MWI': 'Malawi', 
    'MYS': 'Malaysia', 
    'MDV': 'Maldives', 
    'MLI': 'Mali', 
    'MLT': 'Malta', 
    'MHL': 'Marshall Islands', 
    'MTQ': 'Martinique', 
    'MRT': 'Mauritania', 
    'MUS': 'Mauritius', 
    'MYT': 'Mayotte', 
    'MEX': 'Mexico', 
    'FSM': 'Micronesia', 
    'MDA': 'Moldova', 
    'MCO': 'Monaco', 
    'MNG': 'Mongolia', 
    'MNE': 'Montenegro', 
    'MSR': 'Montserrat', 
    'MAR': 'Morocco', 
    'MOZ': 'Mozambique', 
    'MMR': 'Myanmar', 
    'NAM': 'Namibia', 
    'NRU': 'Nauru', 
    'NPL': 'Nepal', 
    'NLD': 'Netherlands', 
    'ANT': 'Netherlands Antilles', 
    'NCL': 'New Caledonia', 
    'NZL': 'New Zealand', 
    'NIC': 'Nicaragua', 
    'NER': 'Niger', 
    'NGA': 'Nigeria', 
    'NIU': 'Niue', 
    'NFK': 'Norfolk Island', 
    'MNP': 'Northern Mariana Islands', 
    'NOR': 'Norway', 
    'OMN': 'Oman', 
    'PAK': 'Pakistan', 
    'PLW': 'Palau', 
    'PSE': 'Palestine', 
    'PAN': 'Panama', 
    'PNG': 'Papua New Guinea', 
    'PRY': 'Paraguay', 
    'PER': 'Peru', 
    'PHL': 'Philippines', 
    'PCN': 'Pitcairn', 
    'POL': 'Poland', 
    'PRT': 'Portugal', 
    'PRI': 'Puerto Rico', 
    'QAT': 'Qatar', 
    'REU': 'Reunion', 
    'ROU': 'Romania', 
    'RUS': 'Russia', 
    'RWA': 'Rwanda', 
    'BLM': 'Saint-Barthelemy', 
    'SHN': 'Saint Helena', 
    'KNA': 'Saint Kitts', 
    'LCA': 'Saint Lucia', 
    'MAF': 'Saint-Martin', 
    'SPM': 'Saint Pierre and Miquelon', 
    'VCT': 'Saint Vincent and Grenadines', 
    'WSM': 'Samoa', 
    'SMR': 'San Marino', 
    'STP': 'Sao Tome and Principe', 
    'SAU': 'Saudi Arabia', 
    'SEN': 'Senegal', 
    'SRB': 'Serbia', 
    'SYC': 'Seychelles', 
    'SLE': 'Sierra Leone', 
    'SGP': 'Singapore', 
    'SVK': 'Slovakia', 
    'SVN': 'Slovenia', 
    'SLB': 'Solomon Islands', 
    'SOM': 'Somalia', 
    'ZAF': 'South Africa', 
    'SGS': 'South Georgia and the South Sandwich Islands', 
    'SSD': 'South Sudan', 
    'ESP': 'Spain', 
    'LKA': 'Sri Lanka', 
    'SDN': 'Sudan', 
    'SUR': 'Suriname', 
    'SJM': 'Svalbard', 
    'SWZ': 'Swaziland', 
    'SWE': 'Sweden', 
    'CHE': 'Switzerland', 
    'SYR': 'Syria', 
    'TWN': 'Taiwan', 
    'TJK': 'Tajikistan', 
    'TZA': 'Tanzania', 
    'THA': 'Thailand', 
    'TLS': 'Timor', 
    'TGO': 'Togo', 
    'TKL': 'Tokelau', 
    'TON': 'Tonga', 
    'TTO': 'Trinidad', 
    'TUN': 'Tunisia', 
    'TUR': 'Turkey', 
    'TKM': 'Turkmenistan', 
    'TCA': 'Turks and Caicos Islands', 
    'TUV': 'Tuvalu', 
    'UGA': 'Uganda', 
    'UKR': 'Ukraine', 
    'ARE': 'United Arab Emirates', 
    'GBR': 'United Kingdom', 
    'USA': 'United States', 
    'UMI': 'US Minor Outlying Islands', 
    'URY': 'Uruguay', 
    'UZB': 'Uzbekistan', 
    'VUT': 'Vanuatu', 
    'VEN': 'Venezuela', 
    'VNM': 'Viet Nam', 
    'VIR': 'Virgin Islands', 
    'WLF': 'Wallis and Futuna', 
    'ESH': 'Western Sahara', 
    'YEM': 'Yemen', 
    'ZMB': 'Zambia', 
    'ZWE': 'Zimbabwe'
}.iteritems())

debates['countries_mentioned'] = debates['token'].apply(lambda token: {x:token.count(x) for x in token if x.encode('utf8') in countries.values()})

#I'll now save this in a table for study and display
country_mentions = pd.concat([debates[["year", "country"]],
                              debates['countries_mentioned'].apply(pd.Series)], axis=1).dropna(axis=1, how='all')
country_mentions['country'] = country_mentions['country'].apply(lambda x: countries.get(x))
country_mentions.head()

#This table is a little too long and large to make sense out of it. Let's group it by year to extract the trend, and by country to get the speaking country info.
country_mentions_by_year = country_mentions.groupby("year").sum()
country_mentions_by_country = country_mentions.groupby("country")[country_mentions.columns[2:]].sum()
#Check the tables
country_mentions_by_year.head()
country_mentions_by_country.head()
#Still not readable... Should order the columns by their sum, then only keep maybe 10?
col_order_year = [k for k in pd.DataFrame({'sum': country_mentions_by_year.sum()}).sort('sum', ascending=False, inplace=False).index]
country_mentions_by_year[col_order_year[0:20]].plot()


# Trying to plot a sankey diagram of countries mentioning each other.
# First need to melt country_mentions_by_country to long form
sankey_data = country_mentions_by_country.unstack().reset_index()

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot
init_notebook_mode()
data = dict(
    type='sankey',
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    valuesuffix = "TWh"   
  )

layout =  dict(
    title = "Which countries mention which in the UN's General Assembly\n(1970-2015)",
    font = dict(
      size = 10
    )
)

data_trace = dict(
    type='sankey',
    width = 1118,
    height = 772,
    domain = dict(
      x =  [0,1],
      y =  [0,1]
    ),
    orientation = "h",
    valueformat = ".0f",
    valuesuffix = "TWh",
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label =  sankey_data['country']
  ),
    link = dict(
      source =  sankey_data['level_0'],
      target =  sankey_data['country'],
      value =  sankey_data[0],
      label =  sankey_data['level_0']
  ))

fig = dict(data=[data_trace], layout=layout)
iplot(fig, validate=False)

#We need to remove the punctuation and english stopwords to only keep the essence of the text.
stop_words = set(stopwords.words('english'))
debates['clean'] = debates['token'].apply(lambda x: [w for w in x if not w in stop_words and not w in punctuation])

#Let's extract the entities (organizations, persons, etc.) speakers are referring to.
debates['entities'] = debates['clean'].apply(lambda x: nltk.ne_chunk(nltk.pos_tag(x), binary=False))
#First example:
debates['entities'][0].draw()
