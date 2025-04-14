import datetime as dt
import experiment1
import ManualStrategy
import experiment2
def author():
    return "lliao32"

def study_group():
    return "lliao32"

if __name__ == "__main__":
    sv = 1000000
    symbol = "JPM"
    insample_sd = dt.datetime(2008, 1, 1)
    insample_ed = dt.datetime(2009, 12, 31)
    outsample_sd = dt.datetime(2010, 1, 1)
    outsample_ed = dt.datetime(2011, 12, 31)

    ms= ManualStrategy.ManualStrategy()
    ms.manual_strategy_in_out_sample_analysis_plotter()

    experiment1.experiment1_plot(symbol, insample_sd, insample_ed, sv, True)
    experiment1.experiment1_plot(symbol, outsample_sd, outsample_ed, sv, False)

    experiment2.experiment2_plot(symbol, insample_sd, insample_ed, sv)