library(shiny)
library(shinydashboard)
library(shinyjs)
library(shinythemes)
library(shinyFiles)
library(dplyr)
library(GGally)
library(ggplot2)
library(ggthemes)
library(corrplot)
library(reshape2)
library(gridExtra)
library(mclust)
library(clustMixType)
library(DescTools)
library(GoodmanKruskal)

############################### ui ###################################
ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "Cardio-disease"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dataset", tabName = "dataset", icon = icon("file")),
      menuItem("EDA", tabName = "eda", icon = icon("chart-bar"))
      #menuItem("Analysis", tabName = "analysis", icon = icon("shapes"))
    )
  ), # sidebar end
  dashboardBody(
    tabItems(
      # First tab content
      tabItem(tabName = "dataset",
              fluidRow(
                box(title = "Input Data(.csv)", width = 3, status = "primary",
                    fileInput("file1", label = NULL)),
                
                box(title = "Data Table", status = "primary", width = 9, solidHeader = T, 
                    div(DT::dataTableOutput("table1"))
                )
              ) # fluidRow end
      ), # tab1 end
      
      # Second tab content
      tabItem(tabName = "eda",
              fluidRow(
                infoBoxOutput("dataset"),
                infoBoxOutput("num_row"),
                infoBoxOutput("num_col")
              ),
              
              fluidRow(
                box(title = "Summary", width = 6, height = 460, verbatimTextOutput("summary")),
                box(title = "Correlation Plot", width = 6, solidHeader = T, plotOutput("corrplot"))
              ),
              
              fluidRow(
                box(title = "Features", width = 6, height = 210,
                    uiOutput("qt"), #呈現給使用者的
                    uiOutput("ql")),
                box(title = "One way ANOVA", width = 6, height = 210, verbatimTextOutput("anova"))
              ),
              
              fluidRow(
                box(title = "Histogram(Quantitative)", width = 4, solidHeader = T, plotOutput("hist_1")),
                box(title = "Boxplot(Quantitative)", width = 4, solidHeader = T, plotOutput("boxplot_1")),
                box(title = "Barplot(Qualitative)", width = 4, solidHeader = T, plotOutput("bar_qual"))
              ),
              
              fluidRow(
                box(title = "Correlation of qualitative variable", width = 4, height = 460, solidHeader = T, verbatimTextOutput("corr_qual")),
                box(title = "Barplot(proportion of var in each cardio type)", width = 4, solidHeader = T, plotOutput("bar_cardio")),
                box(title = "Boxplot(Quantitative X Qualitative)", width = 4, solidHeader = T, plotOutput("boxplot_2"))
              )# fluidRow end
      ) # tab2 end
        
      # # Third tab content
      # tabItem(tabName = "analysis",
      #         # fluidRow(
      #         #       checkboxGroupInput("classifier", choiceNames = c("LDA", "QDA", "Logistic")),
      #         #         textOutput("classifier")
      #         # ), #呈現給使用者的
      # 
      #         fluidRow(
      #           box(title = "Kmean method", width = 6, solidHeader = T, plotOutput("kmeans")),
      #           box(title = "GMM method", width = 6, solidHeader = T, plotOutput("GMM"))
      #         ) # fluidRow end
      # ) # tab3 end
    ) # tabs end
  ) # body end
) # page end

############################### Run APP #######################################
shinyApp(ui, server)

