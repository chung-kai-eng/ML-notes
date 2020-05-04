#0. 下載和開啟packages
install.packages("tidyverse")
library(tidyverse)
install.packages("lubridate")
library(lubridate)


# 1. 下載資料並讀取
url <- "http://ww2.amstat.org/sections/graphics/datasets/DataExpo2018.zip"
temp <- tempfile()
download.file(url, temp)
forecast <- read.table(unz(temp, "forecast.dat"),sep = "" , header = F ,
                       na.strings ="", stringsAsFactors= F)
colnames(forecast) <- c("city.num","date.be.pred","value","item","date.pred")
forecast <- as_tibble(forecast)
unlink(temp)



# 2. 看一下資料、判斷是否為tidy data
View(forecast)
glimpse(forecast)
head(forecast,10)
tail(forecast,20)

#3. 將變數轉為適當的格式
#將兩個date var. 轉成Date格式
forecast$date.be.pred <- ymd(forecast$date.be.pred)
forecast$date.pred <- ymd(forecast$date.pred)
#將value裡的遺漏值"M"，轉成R的格式NA
forecast$value[which(forecast$value=="M")] <- NA
forecast$value <- as.numeric(forecast$value)

#4. 將資料轉換成tidy data
forecast.new <- forecast %>%
        #1.先建立新變項：step，用date.be.pred-date.pred，看是幾步預測
        mutate(step=paste("step",(date.be.pred-date.pred),sep="_")) %>%
        #2.將date.pred拿掉
        select(-date.pred) %>%
        #3.刪除重複值
        distinct(city.num, date.be.pred, item, step, 
                 #這些combination後的unique才留下來
                 .keep_all = TRUE
                 #TRUE表示unique的row，仍要附上上面combination外的變項
        ) %>%
        #4.lag變項中的子類別都變新變項
        spread(step,value) %>%
        #5.排序
        arrange(item,city.num,date.be.pred) %>%
        #6. 把年月日拆開
        separate(date.be.pred,c("year","month","day"),sep="-")

#5.看一下轉換後的結果
View(forecast.new)
glimpse(forecast.new)
head(forecast.new,10)
tail(forecast.new,20)

#6.做點簡單的探索性資料分析
F_to_C <- function(x) {(x - 32) * 5/9} #將溫度由F轉成C

forecast.new %>%
        filter(item=="MinTemp",month=="04") %>%
        mutate(MinTemp_C=F_to_C(step_0)) %>%
        group_by(city.num) %>%
        summarise(N_obs=n(),
                  N_missing=sum(is.na(MinTemp_C)),
                  Mean=mean(MinTemp_C,na.rm=T),
                  SD=sd(MinTemp_C,na.rm=T)) %>%
        arrange(Mean)

