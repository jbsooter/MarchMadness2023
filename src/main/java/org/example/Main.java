package org.example;

import smile.base.cart.Loss;
import smile.base.cart.SplitRule;
import smile.classification.RandomForest;
import smile.data.formula.Formula;
import smile.classification.GradientTreeBoost;
import smile.validation.Accuracy;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.NumberColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.io.csv.CsvReadOptions;

import java.sql.SQLOutput;
import java.util.Arrays;

import static tech.tablesaw.aggregate.AggregateFunctions.*;

public class Main {
    public static void main(String[] args) {
        //read in game data
        Table mGameData = Table.read().csv("data/MRegularSeasonDetailedResults.csv");

        //data after 2022
        mGameData = mGameData.where(mGameData.intColumn("Season").isGreaterThan(2022));

        Table winData =
                mGameData.summarize(mGameData.column(8),
                mGameData.column(9),
                        mGameData.column(10),
                        mGameData.column(11),
                        count,sum).by(mGameData.intColumn("WTeamID"));

        Table lossData = mGameData.summarize(mGameData.column(21),
                mGameData.column(22),
                mGameData.column(23),
                mGameData.column(24),
                count,sum).by(mGameData.intColumn("LTeamID"));

        lossData.column("LTeamID").setName("WTeamID");
        //System.out.println(lossData);

        //sum and count data for wins and losses by team ID
        Table totalData = winData.joinOn("WTeamID").inner(lossData);

        //System.out.println(totalData);

        //create season averages for model
        Table modelData = Table.create(totalData.column("WTeamID").setName("TeamID"));

        modelData.addColumns(
                totalData.numberColumn("Sum [WFGA3]").add(totalData.numberColumn("Sum [LFGA3]")).divide(totalData.numberColumn("Count [WFGA3]").add(totalData.numberColumn("Count [LFGA3]"))).setName("FGA3"));

        modelData.addColumns(
                totalData.numberColumn("Sum [WFGA]").add(totalData.numberColumn("Sum [LFGA]")).divide(totalData.numberColumn("Count [WFGA]").add(totalData.numberColumn("Count [LFGA]"))).setName("FGA"));

        modelData.addColumns(
                totalData.numberColumn("Sum [WFGM3]").add(totalData.numberColumn("Sum [LFGM3]")).divide(totalData.numberColumn("Count [WFGM3]").add(totalData.numberColumn("Count [LFGM3]"))).setName("FGM3"));

        modelData.addColumns(
                totalData.numberColumn("Sum [WFGM]").add(totalData.numberColumn("Sum [LFGM]")).divide(totalData.numberColumn("Count [WFGM]").add(totalData.numberColumn("Count [LFGM]"))).setName("FGM"));


        //System.out.println(modelData);

        //join season averages to games
        Table withOutcomes = Table.create(mGameData.column("WTeamID").copy(),mGameData.column("WTeamID").setName("TeamA"),mGameData.column("LTeamID").setName("TeamB"));

        //System.out.println(withOutcomes);

        for(int i = 0; i < withOutcomes.rowCount();i++)
        {
            if(Math.random() > 0.5)
            {
                withOutcomes.row(i).setInt(1,withOutcomes.row(i).getInt("TeamB"));
                withOutcomes.row(i).setInt(2,withOutcomes.row(i).getInt("WTeamID"));

            }
        }

        IntColumn teamAW = IntColumn.create("TeamAWin");


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

        withOutcomes.addColumns(teamAW);


        //System.out.println(withOutcomes);


        Table finalData = withOutcomes.joinOn("TeamA").inner(modelData,"TeamID");

        for(int i = 1; i < 5;i++)
        {
            modelData.column(i).setName(modelData.column(i).name() + "B");
        }

        finalData = finalData.joinOn("TeamB").inner(modelData,"TeamID");
        //System.out.println(finalData);
        finalData.removeColumns("WTeamID");

        System.out.println(finalData);

        //Split the data 70% test, 30% train
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
        //System.out.println(r64Matchups);

        for(int i = 1; i < 5;i++)
        {
            modelData.column(i).setName(modelData.column(i).name().substring(0,modelData.column(i).name().length()-1));
        }
        r64Matchups = r64Matchups.joinOn("IDTeamA").inner(modelData,"TeamID");
        //System.out.println(r64Matchups);

        for(int i = 1; i < 5;i++)
        {
            modelData.column(i).setName(modelData.column(i).name() + "B");
        }

        r64Matchups = r64Matchups.joinOn("IDTeamB").inner(modelData,"TeamID");
        //System.out.println(r64Matchups);

        r64Matchups.column("IDTeamA").setName("TeamA");
        r64Matchups.column("IDTeamB").setName("TeamB");
        r64Matchups.addColumns(IntColumn.create("TeamAWin"));


        r64Matchups.removeColumns("SeedA","SeedB");

        r64Matchups = Table.create(r64Matchups.column(0),r64Matchups.column(1),r64Matchups.column(10),r64Matchups.column(2),r64Matchups.column(3),r64Matchups.column(4),r64Matchups.column(5),r64Matchups.column(6),r64Matchups.column(7),r64Matchups.column(8),r64Matchups.column(9));
        //get pred

        //predict the response of test dataset with GBModel1
        int[] predictionsR64 = GBModel1.predict(r64Matchups.smile().toDataFrame());

        //System.out.println(Arrays.toString(predictionsR64));
        r64Matchups.removeColumns("TeamAWin");
        r64Matchups.addColumns(IntColumn.create("TeamAWin",predictionsR64));

        //System.out.println(r64Matchups);

        teamKeys.removeColumns("FirstD1Season","LastD1Season");

        Table output = r64Matchups.joinOn("TeamA").inner(teamKeys,"TeamID");
        output.column("TeamName").setName("TeamAName");
        output = output.joinOn("TeamB").inner(teamKeys,"TeamID");
        output.column("TeamName").setName("TeamBName");


        //Write first round predictions
        output.write().csv("Boostingr64Predictions.csv");

    }
}