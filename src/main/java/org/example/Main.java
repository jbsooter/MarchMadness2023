package org.example;

import smile.data.formula.Formula;
import smile.classification.GradientTreeBoost;
import smile.validation.Accuracy;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.HorizontalBarPlot;

import java.util.Arrays;
import java.util.List;

import static tech.tablesaw.aggregate.AggregateFunctions.*;

public class Main {
    public static void main(String[] args) {
        //read in game data
        Table mGameData = Table.read().csv("data/MRegularSeasonDetailedResults.csv");

        //data after 2022
        mGameData = mGameData.where(mGameData.intColumn("Season").isGreaterThan(2022));

        //summarize counts and sums of datapoints for winners and losers
        List<String> wAttributes = List.of("WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF");
        List<String> lAttributes = List.of("LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF");
        Table winData = mGameData.summarize(wAttributes,count,sum).by(mGameData.intColumn("WTeamID"));
        Table lossData = mGameData.summarize(lAttributes,count,sum).by(mGameData.intColumn("LTeamID"));

        //join sum and count measures for wins and losses by same team
        Table totalData = winData.joinOn("WTeamID").inner(lossData,"LTeamID");
        totalData.column("WTeamID").setName("TeamID");

        //create season averages for model
        Table modelData = Table.create(totalData.column("TeamID"));

        for(int i = 0; i < lAttributes.size();i++)
        {
            modelData.addColumns(totalData.numberColumn(String.format("Sum [%s]",wAttributes.get(i))).add(totalData.numberColumn(String.format("Sum [%s]",lAttributes.get(i)))).divide(totalData.numberColumn(String.format("Count [%s]",wAttributes.get(i))).add(totalData.numberColumn(String.format("Count [%s]",lAttributes.get(i))))).setName(wAttributes.get(i).substring(1)));
        }

        //join season averages to games
        Table withOutcomes = Table.create(mGameData.column("WTeamID").copy(),mGameData.column("WTeamID").setName("TeamA"),mGameData.column("LTeamID").setName("TeamB"));

        for(int i = 0; i < withOutcomes.rowCount();i++) {
            if (Math.random() > 0.5) {
                withOutcomes.row(i).setInt(1, withOutcomes.row(i).getInt("TeamB"));
                withOutcomes.row(i).setInt(2, withOutcomes.row(i).getInt("WTeamID"));
            }
        }
        //add column to represent Y
        IntColumn teamAW = IntColumn.create("TeamAWin");

        //mix up team A and B so that each col is a mixture of winners and losers
        for(int i = 0; i < withOutcomes.rowCount();i++)
        {
            if(withOutcomes.row(i).getInt("TeamA") == withOutcomes.row(i).getInt("WTeamID"))
            {
                teamAW.append(1);
            }
            else {
                teamAW.append(0);
            }
        }

        //add AW ahead of join
        withOutcomes.addColumns(teamAW);
        Table finalData = withOutcomes.joinOn("TeamA").inner(modelData,"TeamID");

        //rename columns with B so that they can be joined later
        for(int i = 1; i < modelData.columnCount();i++)
        {
            modelData.column(i).setName(modelData.column(i).name() + "B");
        }

        finalData = finalData.joinOn("TeamB").inner(modelData,"TeamID");
        finalData.removeColumns("WTeamID");


        //TODO: remove columns not desired as features,
        //finalData = finalData.removeColumns("PF","AstB","Stl","DRB","TO","BlkB","OR","ORB","StlB","TOB","FTMB","PFB","Ast","FTA","FTM","DR","Blk","FTAB");

        //Split the data 80% test, 20% train
        Table[] splitData = finalData.sampleSplit(0.8);
        Table dataTrain = splitData[0];
        Table dataTest = splitData[1];

        //Try Gradient Boosting
        GradientTreeBoost GBModel1 = GradientTreeBoost.fit(
                Formula.lhs("TeamAWin"),
                dataTrain.smile().toDataFrame(),
                500, //n
                (int) Math.sqrt((double) (dataTrain.columnCount() - 1)), //m = sqrt(p)
                100, //d
                5,
                0.3,
                1
        );

        //predict the response of test dataset with GBModel1
        int[] predictionsGB = GBModel1.predict(dataTest.smile().toDataFrame());

        //evaluate % classification accuracy for GBModel1
        double accuracyGB = Accuracy.of(dataTest.intColumn("TeamAWin").asIntArray(), predictionsGB);
        System.out.println(accuracyGB);

        //Create data for actual first round games
        Table teamKeys = Table.read().csv("data/MTeams.csv");
        CsvReadOptions.Builder builder =
                CsvReadOptions.builder("data/r64Matchups.csv")
                        .separator('\t')										// table is tab-delimited
                        .header(true)			;								// no header	        ;// the date format to use.

        Table r64Matchups = Table.read().usingOptions(builder);
        r64Matchups.addColumns(IntColumn.create("TeamAWin"));

        //remove B subscripts
        for(int i = 1; i < modelData.columnCount();i++)
        {
            modelData.column(i).setName(modelData.column(i).name().substring(0,modelData.column(i).name().length()-1));
        }
        //add team A stats to r64 table
        r64Matchups = r64Matchups.joinOn("IDTeamA").inner(modelData,"TeamID");

        //add b subscirpts
        for(int i = 1; i < modelData.columnCount();i++)
        {
            modelData.column(i).setName(modelData.column(i).name() + "B");
        }

        //add B stats to table
        r64Matchups = r64Matchups.joinOn("IDTeamB").inner(modelData,"TeamID");

        r64Matchups.column("IDTeamA").setName("TeamA");
        r64Matchups.column("IDTeamB").setName("TeamB");

        r64Matchups.removeColumns("SeedA","SeedB");


        //predict the response for upcoming first cound games
        int[] predictionsR64 = GBModel1.predict(r64Matchups.smile().toDataFrame());

        //print matrix and add to r64 table
        System.out.println(Arrays.toString(predictionsR64));
        r64Matchups.removeColumns("TeamAWin");
        r64Matchups.addColumns(IntColumn.create("TeamAWin",predictionsR64));

        //remove unneeded info
        teamKeys.removeColumns("FirstD1Season","LastD1Season");

        Table output = r64Matchups.joinOn("TeamA").inner(teamKeys,"TeamID");
        output.column("TeamName").setName("TeamAName");
        output = output.joinOn("TeamB").inner(teamKeys,"TeamID");
        output.column("TeamName").setName("TeamBName");


        //Write first round predictions
        output.write().csv("Boostingr64Predictions.csv");

        //check variable importance for current model
        double[] GBModel1_Importance = GBModel1.importance();
        System.out.println(Arrays.toString(GBModel1_Importance));

        //plot variable importance with tablesaw
        Table varImportance = Table.create("featureImportance");
        List<String> featureNames = dataTrain.columnNames();
        featureNames.remove(2); //remove response (TeamAWin)
        varImportance.addColumns(DoubleColumn.create("featureImportance", GBModel1_Importance), StringColumn.create("Feature",  featureNames));
        varImportance = varImportance.sortDescendingOn("featureImportance");
        Plot.show(HorizontalBarPlot.create("Feature Importance", varImportance, "Feature", "featureImportance"));
    }
}