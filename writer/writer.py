# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import datetime as dt
import json
import logging
import os
from collections import Counter
from typing import Dict, List, Optional

import sqlalchemy

from bettingAI.googleCloud.initPostgreSQL import initSession
from bettingAI.googleCloud.databaseClasses import *
from bettingAI.writer.addRow import *
from bettingAI.writer.scraper import *
from bettingAI.writer.values import *


def runner(session: sqlalchemy.orm.Session) -> None:
    """Executes the main logic of the writer program.

    This function updates team and player information, gathers match data, and adds it to the database.
    It iterates over each league, retrieves team links, updates team information, retrieves player links,
    updates player information, retrieves match links, gathers match information, and adds it to the database.
    The function also handles exceptions and logs error messages.

    Args:
        session (sqlalchemy.orm.Session): The database session object used for database operations.

    Returns:
        None
    """
    # Import leagues from database
    leagues = session.query(Leagues).all()
    leagues = [league.__dict__ for league in leagues]
    for league in leagues:  # remove unnecessary keys from the dictionaries
        del league["_sa_instance_state"]

    trackData = (
        Counter()
    )  # class to track explored, updated and added teams, matches and players
    for league in leagues:  # iterate over each league
        logger.info(f"Updating {league['name']}")
        exceptions[league["id"]] = Counter()  # create specific Counter() for this league

        # PART ONE: GET / UPDATE TEAM AND PLAYER INFO
        # Get the fotmob links to all team in that league
        teams = get_team_links(league["id"], nTeams=league["n_teams"])
        teamIDs = [
            row.id
            for row in session.query(Teams.id)
            .filter(Teams.league_id == league["id"])
            .all()
        ]

        for team in teams:  # iterate over each team
            continue
            teamID = int(team.split("/")[2])  # fotmob team id

            if teamID not in teamIDs:  # check if we already have the team in the database
                # Add the team if possible
                try:
                    trackData["tExplored"] += 1
                    teamSQL = add_team(teamID, get_name(team), league["id"])
                    session.add(teamSQL)
                    session.commit()
                    trackData["tAdded"] += 1
                except Exception as e:
                    session.rollback()
                    exceptions[league["id"]][f"{type(e)} : {e}"] += 1

            # Get the fotmob links to all matches and players for the given team
            players = get_player_links(teamID)
            try:
                playerIDs = [
                    row.id
                    for row in session.query(Players.id)
                    .filter(Players.team_id == teamID)
                    .all()
                ]
            except sqlalchemy.exc.DatabaseError as e:
                playerIDs = []

            for player in players:  # iterate over all players in the team
                continue
                playerID = int(player.split("/")[2])
                if playerID in playerIDs:  # check if player already is listed for team
                    continue

                try:
                    trackData["pExplored"] += 1  # update number of explored players
                    playerBio = add_player_bio(
                        player.split("/")[2],
                        teamID,
                        get_name(player),
                        get_player_bio(player),
                    )
                    session.add(playerBio)  # add player
                    session.commit()  # commit player to databse
                    trackData["pAdded"] += 1  # keep track of players added
                except Exception as e:
                    session.rollback()
                    exceptions[league["id"]][f"{type(e)} : {e}"] += 1

        # PART TWO: GET MATCH INFO
        seasons = SEASONS[league["year_span"]]
        seasons = ["2022-2023"] if int(league["year_span"]) == 2 else ["2023"]
        for season in seasons:  # iterate over last ten seasons for the league
            logger.info(f"Begun on {season} season for {league['name']}")
            if league["level"] > 2:
                print("Will not add leagues with lower lever than 2")
                continue

            fixtures = get_match_links(
                league["id"], season
            )  # find all matches in that season
            fixtureIDs = [
                row.id
                for row in session.query(Matches.id)
                .filter((Matches.league_id == league["id"]) & (Matches.season == season))
                .all()
            ]  # find all matches already in the db

            for fixture in fixtures:  # iterate over all matches found for the team
                fixtureID = int(fixture.split("/")[2])
                if fixtureID in fixtureIDs:
                    continue
                # Gather info about that fixture
                try:
                    matchStats, playerStats = get_match_info(fixture)
                    matchID = fixture.split("/")[2]  # get match ID from fotmob
                except Exception as e:
                    exceptions[league["id"]][f"{type(e)} : {e}"] += 1
                    continue

                if (
                    matchStats is False
                ):  # it will return False if the match is in the future
                    continue

                # Add match data to the database
                try:
                    trackData["mExplored"] += 1  # track matches explored
                    match, homeSide, awaySide = add_match(matchID, season, matchStats)
                    session.add(match)  # add main info
                    if homeSide is not None:
                        session.add(homeSide)  # add home stats
                        session.add(awaySide)  # add away stats
                    session.commit()  # commit match stats
                    trackData["mAdded"] += 1  # track match added
                except Exception as e:
                    session.rollback()
                    exceptions[league["id"]][f"{type(e)} : {e}"] += 1

                # Add player performance stats do database
                if playerStats is None:
                    continue
                for player in playerStats.keys():  # iterate over all players returned from gather_player_performance()
                    try:
                        trackData["psExplored"] += 1
                        playerPerformance = add_player_performance(
                            matchID, playerStats, player
                        )
                        session.add(playerPerformance)
                        session.commit()
                        trackData["psAdded"] += 1
                    except Exception as e:
                        session.rollback()
                        exceptions[league["id"]][f"{type(e)} : {e}"] += 1
            print(trackData)

        # Log error messages that occoured in the league
        if exceptions[league["id"]].items():  # check for any exceptions
            logger.info(f"Exceptions for {league['name']}:")
            for error, count in exceptions[
                league["id"]
            ].items():  # iterate over all exceptions
                if (
                    count > 5
                ):  # ensure we dont record the error of trying to add something that is already there
                    logger.error(f"    {error}   ->  {count}")  # log each exception

    print(exceptions)
    # Get the end time of the data gathering
    session.close()
    endTime = dt.datetime.now()

    # Create a report of the writer.py execution
    try:
        createReport(startTime, endTime, trackData)
    except:
        logger.error("Could not create report")
        print(startTime, endTime, trackData)

    # Exit program after successful execution
    logger.info(f"Successfully run in a time of {str(endTime - startTime)}")


def initLogger() -> logging.Logger:
    """Initializes and configures a logger object for logging messages.

    Returns:
        logging.Logger: The logger object configured with file and console handlers.
    """
    # Create a logger object
    logger = logging.getLogger()

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)

    # Create a handler for writing log messages to a file
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"logs/app_{timestamp}.log")
    file_handler.setLevel(logging.INFO)

    # Create a handler for printing log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a formatter for the log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def loadJSON(path: str) -> Dict:
    """
    Loads and returns data from a JSON file located at the specified path.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The data loaded from the JSON file.
    """
    return json.load(open(path))


def createReport(
    start: dt.datetime,
    end: dt.datetime,
    report: Counter,
    test: Optional[bool] = False,
) -> None:
    """
    Creates a report with information about the execution and saves it as a .txt file.

    Args:
        start (datetime.datetime): The start time of the execution.
        end (datetime.datetime): The end time of the execution.
        report (Counter): A Counter object containing various execution statistics.
        test (bool, optional): Specifies whether it is a test run. Defaults to False.

    Returns:
        None: The function does not return any value.
    """
    reportContent = f"""
    -------------------
    Time started: {start}
    Time ended: {end}
    Time delta: {end - start}
    Teams explored: {report.get("tExplored", 0)}
    Teams added: {report.get("tAdded", 0)}
    Matches explored: {report.get("mExplored", 0)}
    Matches added: {report.get("mAdded", 0)}
    Player stats explored: {report.get("psExplored", 0)}
    Player stats added: {report.get("psAdded", 0)}
    Players explored: {report.get("pExplored", 0)}
    Players updated: {report.get("pUpdated", 0)}
    Players added: {report.get("pAdded", 0)}
    -------------------
    """

    # Decide filename
    formattedDate = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"report {formattedDate}.txt"

    # Create and write to file
    if not test:
        with open(os.path.join("reports", filename), "w") as f:
            f.write(reportContent)

    # Confirm that the report was created by returning filename
    logger.info(f"Report saved as {filename}")


if __name__ == "__main__":
    """
    Main entry point of the program.

    The program performs the following steps:
    1. Initializes the logging session and establishes a connection to the PostgreSQL database.
    2. Starts a session with the database.
    3. Executes the main program logic by calling the `runner` function.
    """

    # Start logging session
    exceptions = Counter()  # class to track number of each exception instance
    logger = initLogger()  # initialize logger

    # Register start time of the program
    startTime = dt.datetime.now()

    # Establish connection to the database
    logger.info("Connecting to PostgreSQL")
    try:
        session = initSession()
        logger.info("Connection established")
    except Exception as e:
        logger.critical(f"Could not start session with database. -> {e}")

    # Run program
    runner(session)
