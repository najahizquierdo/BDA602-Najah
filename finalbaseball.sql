CREATE OR REPLACE TABLE last100Games(
  select
    g.game_id
    , g.away_team_id
    , g.home_team_id
    , g.local_date
    , b.winner_home_or_away
  from game g
  JOIN boxscore b ON g.game_id = b.game_id
  WHERE g.local_date BETWEEN DATE_ADD(g.local_date, INTERVAL -100 DAY) AND g.local_date
  group by g.local_date
  order by game_id
);


CREATE OR REPLACE TABLE away_stats_rolling(
    SELECT
        p.game_id as game_id
        , p.team_id as team_id
        , (p.Walk/NULLIF((p.endingInning),0) * 9) AS bb9
        , (p.Home_Run * 9)/NULLIF((p.endingInning),0) AS era
        , (p.Strikeout+p.Walk)/NULLIF((p.endingInning),0) AS pfr
        , (p.Ground_Out+p.Grounded_Into_DP)/NULLIF((p.Fly_Out+p.Sac_Fly),0) AS gf
        , (p.Walk + p.Hit)/NULLIF((p.endingInning),0) AS whip
        , (p.Strikeout/NULLIF((p.endingInning),0) * 9) AS k9
        , b.Hit/NULLIF(b.atBat,0) AS ba
        , b.toBase/NULLIF(b.atBat,0) AS slg
        , (b.Hit/NULLIF(b.atBat,0)) - (b.toBase/NULLIF(b.atBat,0)) as ISO
        , (b.Walk + b.Hit + b.Home_Run)/(NULLIF((b.atBat + b.Walk + b.Hit_By_Pitch + b.Sac_Fly),0)) AS obp
        , b.Grounded_Into_DP AS GDP
    FROM pitcher_counts p
    JOIN batter_counts b ON b.game_id = p.game_id AND b.team_id = p.team_id
    JOIN last100games g ON g.away_team_id = p.team_id and p.game_id = g.game_id
    WHERE EXISTS(SELECT p.game_id FROM last100Games l WHERE l.game_id = p.game_id)
    GROUP BY g.game_id
    );

CREATE OR REPLACE TABLE home_stats_rolling(
    SELECT
        p.game_id as game_id
        , p.team_id as team_id
        , (p.Walk/NULLIF((p.endingInning),0) * 9) AS bb9
        , (p.Home_Run * 9)/NULLIF((p.endingInning),0) AS era
        , (p.Strikeout+p.Walk)/NULLIF((p.endingInning),0) AS pfr
        , (p.Ground_Out+p.Grounded_Into_DP)/NULLIF((p.Fly_Out+p.Sac_Fly),0) AS gf
        , (p.Walk + p.Hit)/NULLIF((p.endingInning),0) AS whip
        , (p.Strikeout/NULLIF((p.endingInning),0) * 9) AS k9
        , b.Hit/NULLIF(b.atBat,0) AS ba
        , b.toBase/NULLIF(b.atBat,0) AS slg
        , (b.Hit/NULLIF(b.atBat,0)) - (b.toBase/NULLIF(b.atBat,0)) as ISO
        , (b.Walk + b.Hit + b.Home_Run)/(NULLIF((b.atBat + b.Walk + b.Hit_By_Pitch + b.Sac_Fly),0)) AS obp
        , b.Grounded_Into_DP AS GDP
    FROM pitcher_counts p
    JOIN batter_counts b ON b.game_id = p.game_id AND b.team_id = p.team_id
    JOIN last100games g ON g.home_team_id = p.team_id and p.game_id = g.game_id
    WHERE EXISTS(SELECT p.game_id FROM last100Games l WHERE l.game_id = p.game_id)
    GROUP BY g.game_id
    );


#league stats for each team???
#home team baseball pythag theorem


#can do based on streaks??



 CREATE OR REPLACE TABLE final_results(
    SELECT
        g.game_id as game_id
        ,g.winner_home_or_away as game_winner
        , CASE WHEN g.winner_home_or_away = 'H' THEN 1
          WHEN g.winner_home_or_away = 'A' THEN 0
          ELSE NULL END AS binary_winner
        ,h.bb9 as home_team_bb9
        ,h.era as home_team_era
        ,h.pfr as home_team_pfr
        ,h.gf as home_team_gf
        ,h.whip as home_team_whip
        ,h.k9 as home_team_k9
        ,h.ba as home_team_ba
        ,h.slg as home_team_slg
        ,h.ISO as home_team_ISO
        ,h.obp as home_team_obp
        ,a.bb9 as away_team_bb9
        ,a.era as away_team_era
        ,a.pfr as away_team_pfr
        ,a.gf as away_team_gf
        ,a.whip as away_team_whip
        ,a.k9 as away_team_k9
        ,a.ba as away_team_ba
        ,a.slg as away_team_slg
        ,a.ISO as away_team_ISO
        ,a.obp as away_team_obp
        from last100Games g
        JOIN home_stats_rolling h ON h.game_id = g.game_id
        JOIN away_stats_rolling a on a.game_id = g.game_id
        GROUP BY game_id
);




