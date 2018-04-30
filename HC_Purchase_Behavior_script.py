
# coding: utf-8
#import importlib
#progressbar_lib = importlib.util.find_spec("progressbar")
#found = (progressbar_lib is not None)
#if found:
#	import progressbar as pb

import pandas as pd
import numpy as np
import re
from functools import reduce
import os
import time, sys

start = time.time()

def update_progress(job_title, progress):
    length = 30 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

print('Welcome to HC QuickBooks Purchase Data Processing Script \n')

print('\n Note that the input file must have at least the following columns \n Date, Transaction Type, Product/Service, Memo/Description, Qty, Amount \n')

f_input = input('Enter the input file name with .csv (exported from QuickBooks reports named \"Sales by Class Detail\"): ')


if f_input is None or f_input == '':
	print('You didn\'t enter a file name. Using \'sales_by_class.csv\' as the default file name')
	f_input = 'sales_by_class.csv'

sales_data = pd.read_csv(f_input, skiprows=4)

update_progress("Processing data...", 0.1)

sales_data.rename(columns={'Unnamed: 0': 'Customer_Name'}, inplace = True)

update_progress("Processing data...", 0.2)

sales_data['Customer_Name'] = sales_data['Customer_Name'].fillna(method='ffill')

update_progress("Processing data...", 0.3)

sales_data = sales_data[~sales_data['Customer_Name'].str.contains("Total for")]


sales_data = sales_data.dropna(axis = 0, subset=['Date'])


sales_data['Date'] = pd.to_datetime(sales_data['Date'])



# #### drop the Sales and Customer columns



sales_data = sales_data.drop(['Customer','Sales Price'], axis = 1)



# ### Change the data type of sales and amount columns




def clean_amount_sales(x):
    
    if isinstance(x, str):
        x = x.strip().replace(",","")
    return x
    
sales_data['Amount'] = (sales_data['Amount'].apply(clean_amount_sales))
#sales_data['Sales Price'] = (sales_data['Sales Price'].apply(clean_amount_sales))

sales_data['Amount'] = pd.to_numeric(sales_data['Amount'], errors = 'raise')
#pd.to_numeric(sales_data['Sales Price'], errors = 'raise')



#sales_data.to_csv('purchase_data_cleaned.csv', sep=',', encoding='utf-8')



## Very useful operations

#sales_data[sales_data['Product/Service'].isnull() & (sales_data['Memo/Description'].str.contains('Shipping') == False)]
#sales_data[sales_data['Product/Service'].isnull()]['Memo/Description'].str.contains('Shipping')

sales_discount = sales_data[(~sales_data['Memo/Description'].isnull() & (sales_data['Memo/Description'].str.contains('Discount')))]


# ### aggregate discounts across customers

update_progress("Processing data...", 0.4)



customer_total_discount = sales_discount.groupby('Customer_Name').agg({'Amount': sum})





customer_total_discount_no_refund = sales_discount[sales_discount.Amount<0].groupby('Customer_Name').agg({'Amount': sum})





##################################################
customer_total_discount_no_refund.rename(columns={'Amount': 'Total Discount without Refund'}, inplace = True)


update_progress("Processing data...", 0.5)



############################
customer_total_discount.rename(columns={'Amount': 'Total Discount with Refund'}, inplace = True)





#customer_total_discount.to_csv('customer_total_discounts.csv', sep=",")
#customer_total_discount_no_refund.to_csv('customer_total_discounts_no_refund.csv', sep=",")


# #### calculate total refund and shipping

update_progress("Processing data...", 0.6)



sales_refund_no_shipping_discount = sales_data[(~sales_data['Transaction Type'].isnull()) & (sales_data['Memo/Description'].str.contains('Discount|discount') == False) & (sales_data['Memo/Description'].str.contains('Shipping|shipping') == False) & (sales_data['Transaction Type'].str.contains('refund|Refund') == True)]
sales_shipping = sales_data[(~sales_data['Memo/Description'].isnull()) & (sales_data['Memo/Description'].str.contains('Shipping|shipping')) == True]
sales_no_shipping = sales_data[(~sales_data['Memo/Description'].isnull()) & (sales_data['Memo/Description'].str.contains('Shipping|shipping')) == False]
sales_refund_no_shipping_w_discount = sales_data[(~sales_data['Transaction Type'].isnull()) & (sales_data['Memo/Description'].str.contains('Shipping|shipping') == False) & (sales_data['Transaction Type'].str.contains('refund|Refund') == True)] 

update_progress("Processing data...", 0.7)




## the code below drops NaN values in Amount, which usually represent shipping refunds
sales_refund_no_shipping_discount = sales_refund_no_shipping_discount.dropna(axis = 0, subset = ['Amount'])
sales_refund_no_shipping_w_discount = sales_refund_no_shipping_w_discount.dropna(axis = 0, subset = ['Amount'])


# ### aggregate amount feature


update_progress("Processing data...", 0.8)


t = sales_data.groupby(['Customer_Name'], group_keys = False).agg({'Amount': sum}).sort_values(by='Amount',ascending=False)





t_shipping = sales_shipping.groupby(['Customer_Name']).agg({'Amount':sum})





t_sales_refund_no_ship_disc = sales_refund_no_shipping_discount.groupby(['Customer_Name'], group_keys = False).agg({'Amount':sum})





t_sales_refund_no_ship_w_disc = sales_refund_no_shipping_w_discount.groupby(['Customer_Name'], group_keys = False).agg({'Amount':sum})





t_shipping.rename(columns={'Amount': 'Total Shipping with Refund and Discount'}, inplace = True)


update_progress("Processing data...", 0.9)



t_sales_refund_no_ship_disc.rename(columns={'Amount': 'Total Refund without Shipping and Discount'}, inplace = True)


update_progress("Processing data...", 1)



#t_sales_refund_no_ship_w_disc





#t.to_csv('test.csv', sep = ',', encoding = 'utf-8')
#t_shipping.to_csv('sales_only_shipping.csv', sep = ',', encoding = 'utf-8')
#t_sales_refund_no_ship_disc.to_csv('sales_refund_no_ship_disc.csv', sep = ',', encoding = 'utf-8')





##########################################
#t_shipping
#t_sales_refund_no_ship_w_disc


# ### Get the date of first purchase




sales_data = sales_data.set_index(['Customer_Name', 'Date'])
#sales_data.reset_index()


update_progress("Writing tables...", 0.1)


#t1 = sales_data.groupby(['Customer_Name', 'Date'], group_keys = True).sum()





s1 = sales_data.reset_index()





## Something interesting
# t1 = sales_data[['Customer_Name', 'Date', 'Amount']].groupby(['Customer_Name', 'Date'], group_keys = True).apply(lambda x: x.sort_values('Date', ascending = True)).groupby(level=[0,1]).head(3)
# t1





# using aggregation
t1 = sales_data.groupby(['Customer_Name', 'Date'], group_keys = True).agg({'Amount': sum})



update_progress("Writing tables...", 0.2)

#t1 = t1.sort_index(level = ['Date'], ascending = [True])





#t1.index.get_level_values(level=0).unique()
## Success!
#t1.to_csv('t1.csv', sep = ",", encoding = "utf-8")





customer_first_purchase = t1.reset_index().groupby(['Customer_Name'], group_keys = True).apply(lambda x: x[['Date', 'Amount']].iloc[0])





customer_first_purchase.rename(columns={'Amount': 'Amount in 1st Sale'}, inplace = True)





customer_first_purchase.rename(columns={'Date': 'Date of 1st Sale'}, inplace = True)



update_progress("Writing tables...", 0.4)

################################
#customer_first_purchase
## Write to a file
#customer_first_purchase.to_csv('First_purchase_total.csv')


# ### get the date of last purchase




customer_latest_purchase = t1.reset_index().groupby(['Customer_Name'], group_keys = True).apply(lambda x: pd.Series(x[['Date', 'Amount']].iloc[-1]))





customer_latest_purchase.rename(columns={'Date': 'Date of last Sale', 'Amount': 'Amount in last Sale'}, inplace = True)


update_progress("Writing tables...", 0.6)


########################################
#customer_latest_purchase





#customer_latest_purchase.to_csv('Latest_purchase_total.csv', sep = ",", encoding = 'utf-8')





#t2 = t1.groupby('Customer_Name').sum()


# #### get total quantity of each purchase




sales_data = sales_data.reset_index()





#sales_data = sales_data.drop(['index'], axis = 1)





#sales_data


# ### sales_item_only: excludes discounts, refunds, shipping




sales_item_only = sales_data[(~sales_data['Product/Service'].isnull()) & (sales_data['Product/Service'].str.contains('Discount') == False) & (sales_data['Memo/Description'].str.contains('Shipping|shipping') == False) & (sales_data['Transaction Type'].str.contains('refund|Refund') == False)]


# ### sales_item_only: excludes discounts, refunds, shipping, credit memo for calculating total amount


update_progress("Writing tables...", 0.8)

sales_item_only_no_credit = sales_item_only[~(sales_item_only['Transaction Type'].str.lower().str.contains('credit memo'))]





t3 = sales_item_only_no_credit.groupby(['Customer_Name']).agg({'Qty': sum})



update_progress("Writing tables...", 1)

#sales_item_only





#sales_item_only_no_credit.to_csv('sales_item_only_no_credit.csv', sep = ",", encoding = "utf-8")


# ### if you find "micro ring" and "links" in the Description = qty/250 AND if you find "SERVICES" in Product Type = 1
# ### if the quantity is in multiple of 250, then get the quotient
# ### use the divmod operator




#p = sales_item_only_no_credit['Memo/Description'].str.lower().str.contains('cylinder links') | sales_item_only['Memo/Description'].str.lower().str.contains('links') | sales_item_only['Memo/Description'].str.lower().str.contains('rings') | sales_item_only['Memo/Description'].str.lower().str.contains('copper') | sales_item_only['Product/Service'].str.lower().str.contains('service')





#sales_item_only[~p].to_csv('temp_cylinder.csv', sep = ",", encoding = "utf-8")


# ### working now! 4/28

update_progress("Filtering quantities...", 0.1)


def filter_quantity(row):
    
    #row['compressed_quantity'] = 0
    compressed_qty = 0
    missed_qty = []
    for i, r in row.iterrows():
         
        if ('copper links' in r['Memo/Description'].lower()) |         ('cylinder links' in r['Memo/Description'].lower()) |         ('copper ring' in r['Memo/Description'].lower()) |         ('services' in r['Product/Service'].lower()) |         ('micro ring' in r['Memo/Description'].lower()):
        
            #t_qty = r['Qty']
            
            if 'services' in r['Product/Service'].lower():
                compressed_qty += 1
            
            elif (divmod(r['Qty'], 250)[1] == 0) & (r['Qty'] > 200):
                compressed_qty += divmod(r['Qty'], 250)[0]
            
            elif (re.search('^cylinder copper link', r['Memo/Description'], flags=re.IGNORECASE))             is not None:
                  compressed_qty += r['Qty']
                  
            else:
                compressed_qty += r['Qty']
                #missed_qty.append(str(r['Memo/Description'])+'')
        else:
                compressed_qty += r['Qty']
            
            
    #return pd.Series({'Compressed Qty': compressed_qty, 'Missed Qty': missed_qty})
    return pd.Series({'Compressed Qty': compressed_qty, 'Normal Qty': sum(row['Qty']), 'Amount': sum(row['Amount'])})   



update_progress("Filtering quantities...", 0.5)

t31 = sales_item_only_no_credit.groupby(['Customer_Name']).apply(filter_quantity)


# #### t31 is filtered


update_progress("Filtering quantities...", 1)

#t31.dropna(axis=0).to_csv('temp_temp.csv', sep = ',', encoding = 'utf-8')
#t31.to_csv('temp_temp.csv', sep = ',', encoding = 'utf-8')





#t3 = sales_item_only_no_credit.groupby(['Customer_Name']).agg({'Qty': filter_quantity, 'Amount': sum})





## t31 when done
t31.rename(columns={'Amount': 'Total Amount in Sales excluding shipping, discount, refund, credit memo', 'Qty': 'Total quantity of purchases excluding shipping, discount, refund, credit memo, filtered '}, inplace=True)





############################
#t31





#t3.to_csv('sales_quantity_no_ship_refund_discount.csv', sep = ',', encoding = 'utf-8')


# #### get the average date intervals between purchases and times order placed

update_progress("Processing dates...", 0.1)


def cal_date_avg(row):
    
    m_date = row['Date']
    m_date = (m_date.unique())
    tmp_days = list()
    #m_indices = m_date.index.values
    
    if len(m_date) == 1:
        avg_days = 0
        #return 0
    else:        
        for i in range(0, len(m_date)):
            if i == 0:
                continue
                
            tmp_days.append((m_date[i] - m_date[i-1]).astype('timedelta64[D]').astype(int))
    
        avg_days = sum(tmp_days)/len(tmp_days)
    row['avg_days'] = avg_days
    #row['count'] = len(m_indices)
    #return m_date.index.values  
    return pd.Series([avg_days, len(m_date)], index = ['avg_days', 'order_count_days'])



update_progress("Processing dates...", 0.2)


t5 = sales_item_only[['Customer_Name', 'Date']].groupby(['Customer_Name'], as_index = True).apply(cal_date_avg)#.to_frame()



update_progress("Processing dates...", 0.5)


t5.rename(columns={'avg_days': 'Avg. Time per Order', 'order_count_days': 'Times Ordered'}, inplace=True)





######################################
#t5





#t5.to_csv('avg_days_purchase_count_days.csv', sep = ',', encoding = 'utf-8')


# ### get % of each application type and top colors and lengths




sales_item_only = sales_item_only.set_index('Customer_Name')



update_progress("Processing dates...", 1)


#sales_item_only



update_progress("Getting application types...", 0.1)


def get_app_type(row):
    
    poly_count = 0
    cylinder_count = 0
    weft_count = 0
    other_count = 0
    
    layered_count = 0
    curly_count = 0
    premium_count = 0
    
    total_color_count = 0
    total_len_count = 0
    '''   
    len_12 = 0
    len_14 = 0
    len_16 = 0
    len_18 = 0
    len_20 = 0
    len_22 = 0
    len_22_plus = 0
    '''
    
    len_exp = re.compile(r'\d{1,2}[ANBGR]\/?\s(\d{2})\s?[SL]?') #len_exp.search(r['Product/Service]).group(1)
    color_exp = re.compile(r'(\d{1,2}[ANBGR]\/?\d?\d?[ANBGR]?)') #re.compile(r'(\d{1,2})[ANBGR]\/?')
    
    m_desc = row[['Memo/Description', 'Product/Service', 'Qty']]
    #return m_desc['Memo/Description'].iloc[0]
    #return row.count()
    
    #return m_desc.apply(lambda x: type(x))
    
    color_dict = dict()
    len_dict = dict()
    # debug
    #product_list = []
    #other_list = []

    for i, r in m_desc.iterrows():
        
        if (((not 'ionix' in r['Memo/Description'].lower()) | (not 'ionix' in r['Product/Service'].lower())) & (not 'color ring' in r['Product/Service'].lower()) & (not 'copper link' in r['Memo/Description'].lower())& (not 'SERVICES' in r['Product/Service']) & (not 'cylinder copper link' in r['Memo/Description'].lower()) & (not 'cylinder link' in r['Memo/Description'].lower())):
            
            if ('poly' in r['Memo/Description'].lower())             & (not 'poly remover' in r['Product/Service'].lower()) & (not 'SERVICES' in r['Product/Service']):
                poly_count += r['Qty']
            
            if ('wefts' in r['Memo/Description'].lower()) & (not 'SERVICES' in r['Product/Service']):
                weft_count += r['Qty']
            # Done: @vishal: make sure that cylinder copper link and cylinder links are not counted here
            if ('cylinder' in r['Memo/Description'].lower()) & (not 'SERVICES' in r['Product/Service']):
                cylinder_count += r['Qty']
                # debug
                #product_list.append(r['Memo/Description'])
                
            if 'layered' in r['Memo/Description'].lower():
                layered_count += r['Qty']
                
            if 'curly' in r['Memo/Description'].lower():
                curly_count += r['Qty']
                
            if 'premium' in r['Memo/Description'].lower():
                premium_count += r['Qty']
                
            if len_exp.search(r['Product/Service']) is not None:
                tmp_val = len_exp.search(r['Product/Service']).group(1)
                total_len_count += r['Qty']
                
                if not tmp_val in len_dict:
                    len_dict[tmp_val] = r['Qty']
                else:
                    len_dict[tmp_val] += r['Qty']
                    
            if color_exp.search(r['Product/Service']) is not None:
                tmp_val = color_exp.search(r['Product/Service']).group(1)
                total_color_count += 1
                
                if not tmp_val in color_dict:
                    color_dict[tmp_val] = r['Qty']
                else:
                    color_dict[tmp_val] += r['Qty']
                    
        else:
        	# @vishal: the total uncompressed quantity
            other_count += r['Qty']
            # debug
            #other_list.append(r['Memo/Description'])
        
    # sort the dictionaries
        
    color_tuples = sorted(color_dict.items(), key = lambda x: x[1], reverse = True)
    len_tuples = sorted(len_dict.items(), key = lambda x: x[1], reverse = True)
        
        #return row
        #row['poly_count'] = poly_count
        #row['weft_count'] = weft_count
        #row['cylinder_count'] = cylinder_count
        #row['other_count'] = other_count
           
        #print(((color_tuples)))
        #print(len(color_tuples))
        #print(r['Product/Service'])
        
    m_series = pd.Series({'poly_count': poly_count, 'weft_count': weft_count, 'cylinder_count': cylinder_count, 'other_count': other_count, 'curly_count': curly_count, 'premium_count': premium_count, 'layered_count': layered_count, 'length_12': 0, 'length_14': 0, 'length_16': 0, 'length_18': 0, 'length_20': 0, 'length_22': 0,  'length_22+': 0})
    
    # get %12	%14"	%16"	%18"	%20"	%22"	%22"+
  
    for lt in len_tuples:
        lt_index = lt[0]
        lt_count = lt[1]
        #len_str = 'length_'
        
        if lt_index == '12':
            #len_12 = lt_count
            m_series = m_series.set_value('length_12', lt_count)
        #else:
            #m_series = m_series.set_value('length_12', 0)
            
        if lt_index == '14':
            #len_14 = lt_count
            m_series = m_series.set_value('length_14', lt_count)
        #else:
            #m_series = m_series.set_value('length_14', 0)
            
        if lt_index == '16':
            #len_16 = lt_count
            m_series = m_series.set_value('length_16', lt_count)
        #else:
            #m_series = m_series.set_value('length_16', 0)
                
        if lt_index == '18':
            #len_18 = lt_count
            m_series = m_series.set_value('length_18', lt_count)
        #else:
            #m_series = m_series.set_value('length_18', 0)
                
        if lt_index == '20':
            #len_20 = lt_count
            m_series = m_series.set_value('length_20', lt_count)
        #else:
            #m_series = m_series.set_value('length_20', 0)
            
        if lt_index == '22':
            #len_22 = lt_count
            m_series = m_series.set_value('length_22', lt_count)
        #else:
            #m_series = m_series.set_value('length_22', 0)
            
        if int(lt_index) > 22:
            m_series = m_series.set_value('length_22+', lt_count)
        #else:
            #m_series = m_series.set_value('length_22+', 0)
    
    
    # get the top 5 colors and their counts
    j = 0
    while j < 5:
         
        col_tmp_str = 'color'+'_'    
        col_tmp_str += str(j+1)
            
        if (j > len(color_tuples) - 1) | (color_tuples is None) | (len(color_tuples) == 0):
            #row[col_tmp_str] = -1
            #row[col_tmp_str+'_'+'count'] = -1 
            m_series = m_series.set_value(col_tmp_str, 0)
            m_series = m_series.set_value(col_tmp_str+'_'+'count', 0)
            
        else:   
            #row[col_tmp_str] = ((color_tuples[j][0]))
            m_series = m_series.set_value(col_tmp_str, (color_tuples[j][0]))
            m_series = m_series.set_value(col_tmp_str+'_'+'count', (color_tuples[j][1]))
            #row[col_tmp_str+'_'+'count'] = color_tuples[j][1]
        j += 1

    row['total_color'] = total_color_count
    row['total_len'] = total_len_count
    
    m_series = m_series.set_value('total_color', total_color_count)
    m_series = m_series.set_value('total_len', total_len_count)
    # debug
    #m_series = m_series.set_value('product_list_cylinder', product_list)
    #m_series = m_series.set_value('other_list', other_list)
    return m_series

    '''
    # tmp solution
    tmp_k = 0
    tmp_v = 0
    if (len(color_tuples) == 0) | (color_tuples is None):
        color_tuples = None
    else:
        tmp_k = color_tuples[0][0]
        tmp_v = color_tuples[0][1]
    '''
    #return count
    
    #if row['Memo/Description'].str.contains('Poly Remover').any(): #and row['Memo/Description'].str.contains('Poly Remover') == False:
    #    return row['Memo/Description']
    #else:
    #    return "holy"
    


update_progress("Getting application types...", 0.5)


t6 = sales_item_only[['Memo/Description', 'Product/Service', 'Qty']].groupby(['Customer_Name'], as_index = True).apply(get_app_type)#.to_frame()#apply(lambda x: x['Memo/Description'].str.contains('Poly Remover')).to_frame()


update_progress("Getting application types...", 1)


#################################
#t6





#t6.to_csv('app_type_and_color_len.csv', sep = ',', encoding = 'utf-8')


# ## Merge Everything

update_progress("Merging all tables...", 0.1)


dfs = [customer_first_purchase, customer_total_discount, customer_total_discount_no_refund, t_shipping, t_sales_refund_no_ship_disc, customer_latest_purchase, t31, t5, t6]


# #### trying to manually merge everything

update_progress("Merging all tables...", 0.2)

df_merged = reduce(lambda l_df, r_df: pd.merge(l_df, r_df, how = 'left', left_index = True, right_index = True), dfs)


update_progress("Merging all tables...", 1)

update_progress("Writing file...", 0.5)

time_string = (pd.to_datetime('now')).strftime("%m-%d-%Y_%H-%M")
file_name = 'HC_purchase_data_export_'+time_string+'.csv'
file_name1 = 'HC_purchase_data_cleaned_'+time_string+'.csv'
out_dir = '.\\purchase_data_exports'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

full_path = os.path.join(out_dir, file_name)
full_path1 = os.path.join(out_dir, file_name1)

df_merged.to_csv(full_path, sep = ",", encoding = "utf-8")

sales_data.to_csv(full_path1, sep=',', encoding='utf-8')

update_progress("Writing files...", 1)

end = time.time()

print("The script took "+str(round(end - start,2))+" seconds to finish. \n If you had any issues, please email vishalvatnani@gmail.com regardless of where he may be right now!")

#pd.merge(customer_first_purchase, customer_total_discount, how = 'left', left_index = True, right_index = True)





#from functools import reduce
#df_merged = reduce(lambda  left,right: pd.merge(left = 'Customer_Name', right = 'Customer_Name',
#                                            how='inner'), dfs)

