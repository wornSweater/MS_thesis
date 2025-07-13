# original features of the oneDTE dataset
FEATURES_ORIGINAL = [
'secid',
'date',
'symbol',
'symbol_flag',
'exdate',
'last_date',
'cp_flag',
'strike_price',
'best_bid',
'best_offer',
'volume',
'open_interest',
'impl_volatility',
'delta',
'gamma',
'vega',
'theta',
'optionid',
'cfadj',
'am_settlement',
'contract_size',
'ss_flag',
'forward_price',
'expiry_indicator',
'root',
'suffix',
'cusip',
'ticker',
'sic',
'index_flag',
'exchange_d',
'class',
'issue_type',
'industry_group',
'issuer',
'div_convention',
'exercise_style',
'am_set_flag'
]



# general features of each field
FEATURES_GREEKS = [
    'delta', 
    'gamma', 
    'vega', 
    'theta',
]

FEATURES_IMPLIED_VOL = [
    'impl_volatility'
]

FEATRUES_BASIC_OPTION = [
    'best_bid', 
    'best_offer', 
    'volume',
    'open_interest', 
    'cp_flag_num', 
    'am_settlement_num', 
    'strike', 
    'option_price',
    'scaled_option_price', 
    'scaled_best_bid',
    'scaled_best_offer',
    'intrinsic_value', 
    'moneyness',
    'log_moneyness', 
    'bid_ask_spread',
    'bid_ask_spread_percentage', 
]

FEATRUES_BASIC_OPTION_NO_INDICATORS = [
    'best_bid', 
    'best_offer', 
    'volume',
    'open_interest', 
    'strike', 
    'option_price',
    'scaled_option_price', 
    'scaled_best_bid',
    'scaled_best_offer',
    'intrinsic_value', 
    'moneyness',
    'log_moneyness', 
    'bid_ask_spread',
    'bid_ask_spread_percentage', 
]

FEATURES_VOLUME_OPEN_INTEREST = [
    'call_total_volume', 
    'put_total_volume', 
    'total_volume', 
    'call_total_open_interest', 
    'put_total_open_interest', 
    'total_open_interest'
]

FEATURES_INTEREST_RATE = [
    'rf', 
    'rf_annual',
]

FEATURES_FORWARD_PRICE = [
    'filled_forward_price',
]

FEATURES_STOCK = [
    'stock_low_price',
    'stock_high_price', 
    'stock_open_price', 
    'stock_close_price',
    'stock_return', 
]

FEATURES_VIX = [
    'vixo',
    'vixh', 
    'vixl', 
    'vix'
]

FEATURES_HISTORICAL_VOL = [
    'vol_days_10', 
    'vol_days_14',
    'vol_days_30', 
    'vol_days_60', 
    'vol_days_91', 
    'vol_days_122',
    'vol_days_152', 
    'vol_days_182', 
    'vol_days_273', 
    'vol_days_365',
    'vol_days_547', 
    'vol_days_730', 
    'vol_days_1825',
]

FEATURES_FAMA = [
    'mktrf', 
    'smb', 
    'hml', 
    'rmw', 
    'cma', 
    'umd',
]

FEATURES_DATE = [
    'date',
    'exdate',
    'datetime',
]

FEATURES_YEAR_MONTH_DAY = [
    'year',
    'month',
    'day',
]

FEATURES_OTHER = [
    'secid',
    'symbol',
    'symbol_flag',
    'last_date',
    'cp_flag',
    'strike_price',
    'optionid',
    'cfadj',
    'am_settlement',
    'contract_size',
    'ss_flag',
    'forward_price',
    'expiry_indicator',
    'root',
    'suffix',
    'cusip',
    'ticker',
    'sic',
    'index_flag',
    'exchange_d',
    'class',
    'issue_type',
    'industry_group', 
    'issuer', 
    'div_convention', 
    'exercise_style',
    'am_set_flag', 
]

FEATURES_RETURN = [
    'payoff',
    'option_return',
    'best_bid_return',
    'best_offer_return',
]

FEATURES_SETTLEMENT_INFO = [
    'settlement_value', 
    'settlement_symbol', 
    'starting_code',
]

FEATURES_WHOLE = [
    'secid', 
    'date', 
    'symbol', 
    'symbol_flag', 
    'exdate', 
    'last_date',
    'cp_flag', 
    'strike_price', 
    'best_bid', 
    'best_offer', 
    'volume',
    'open_interest', 
    'impl_volatility', 
    'delta', 
    'gamma', 
    'vega', 
    'theta',
    'optionid', 
    'cfadj', 
    'am_settlement', 
    'contract_size', 
    'ss_flag',
    'forward_price', 
    'expiry_indicator', 
    'root', 
    'suffix', 
    'cusip',
    'ticker', 
    'sic', 
    'index_flag', 
    'exchange_d', 
    'class', 
    'issue_type',
    'industry_group', 
    'issuer', 
    'div_convention', 
    'exercise_style',
    'am_set_flag', 
    'settlement_value', 
    'settlement_symbol', 
    'starting_code',
    'cp_flag_num', 
    'am_settlement_num', 
    'strike', 
    'payoff', 
    'option_price',
    'scaled_option_price', 
    'scaled_best_bid',
    'scaled_best_offer',
    'option_return', 
    'best_bid_return',
    'best_offer_return',
    'stock_low_price',
    'stock_high_price', 
    'stock_open_price', 
    'stock_close_price',
    'stock_return', 
    'intrinsic_value', 
    'moneyness', 
    'log_moneyness',
    'bid_ask_spread',
    'bid_ask_spread_percentage', 
    'vol_days_10', 
    'vol_days_14',
    'vol_days_30', 
    'vol_days_60', 
    'vol_days_91', 
    'vol_days_122',
    'vol_days_152', 
    'vol_days_182', 
    'vol_days_273', 
    'vol_days_365',
    'vol_days_547', 
    'vol_days_730', 
    'vol_days_1825', 
    'filled_forward_price',
    'mktrf', 
    'smb', 
    'hml', 
    'rmw', 
    'cma', 
    'umd', 
    'rf', 
    'rf_annual', 
    'vixo',
    'vixh', 
    'vixl', 
    'vix',
    'call_total_volume', 
    'put_total_volume', 
    'total_volume', 
    'call_total_open_interest', 
    'put_total_open_interest', 
    'total_open_interest',
    'datetime',
    'year',
    'month',
    'day',
]



# ATM option features
FEATURES_CALL_ATM = [
    'call_time_value',
    'log_call_time_value',
]

FEATURES_PUT_ATM = [
    'put_time_value',
    'log_put_time_value',
]

FEATURES_STRADDLE_PUT = [
    'put_delta',
    'put_gamma',
    'put_vega',
    'put_theta',
    'put_impl_volatility',
    'put_best_bid',
    'put_best_offer',
    'put_volume',
    'put_open_interest',
    'put_option_price',
    'put_scaled_option_price',
    'put_scaled_best_bid',
    'put_scaled_best_offer',
    'put_intrinsic_value',
    'put_bid_ask_spread',
    'put_bid_ask_spread_percentage',
    'put_time_value',
    'log_put_time_value',
 ]

FEATURES_STRADDLE_PUT_PAYOFF = [
    'put_delta',
    'put_gamma',
    'put_vega',
    'put_theta',
    'put_impl_volatility',
    'put_best_bid',
    'put_best_offer',
    'put_volume',
    'put_open_interest',
    'put_option_price',
    'put_scaled_option_price',
    'put_scaled_best_bid',
    'put_scaled_best_offer',
    'put_intrinsic_value',
    'put_bid_ask_spread',
    'put_bid_ask_spread_percentage',
    'put_time_value',
    'log_put_time_value',
    'put_payoff',
 ]

FEATURES_STRADDLE_EXTRA = [
    'implied_vol_diff',
    'average_implied_vol',
    'volume_summation',
    'open_interest_summation',
    'average_bid_ask_spread',
    'average_bid_ask_spread_percentage',
    'vol_days_10_minus_implied_vol',
    'vol_days_14_minus_implied_vol',
    'vol_days_30_minus_implied_vol',
    'vol_days_60_minus_implied_vol',
    'vol_days_91_minus_implied_vol',
    'vol_days_122_minus_implied_vol',
    'vol_days_152_minus_implied_vol',
    'vol_days_182_minus_implied_vol',
    'vol_days_273_minus_implied_vol',
    'vol_days_365_minus_implied_vol',
    'vol_days_547_minus_implied_vol',
    'vol_days_730_minus_implied_vol',
    'vol_days_1825_minus_implied_vol',
    'total_price',
    'total_best_offer',
    'total_best_bid',
 ]

FEATURES_STRADDLE_RETURN = [
    'straddle_return',
    'straddle_offer_return',
    'straddle_bid_return',
]



# auxiliary features
# indicator features that do not need to be standardized
FEATURES_INDICATOR = [
    'cp_flag_num', 
    'am_settlement_num', 
]

FEATURES_WHOLE_SCALED = (
    FEATRUES_BASIC_OPTION_NO_INDICATORS + 
    FEATURES_VOLUME_OPEN_INTEREST + 
    FEATURES_INTEREST_RATE + 
    FEATURES_FORWARD_PRICE + 
    FEATURES_STOCK + 
    FEATURES_VIX +
    FEATURES_HISTORICAL_VOL +
    FEATURES_FAMA
)