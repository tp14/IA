import streamlit as st
from board import *
import pygame

if __name__ == "__main__":
    
    pygame.init()

    title = st.markdown("<h1 style='font-weight: bold; text-align: center;'>IA - Ant Clustering</h1>", unsafe_allow_html=True)

    homo, divider, hete = st.columns([1, 0.05, 1])

    with homo:
        homo_title = st.markdown("<h2 style='color: teal;'>Homogeneous Clustering</h2>", unsafe_allow_html=True)
        homo_ants = st.number_input("How Many Ants:", min_value=1, max_value=50, value=10, key="homo_ants")
        ant_amount = st.number_input("Quantity of food:", min_value=5, max_value=1000, value=75, key="homo_food")
        dimension = st.number_input("Space size:", min_value=5, max_value=50, value=20, key="dimension")

        st.markdown(
            f"""
            <style>
            .homo-button {{
                background-color: teal;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
            }}
            </style>
            """
            , unsafe_allow_html=True
        )
        CORPSES_NUM = 100

        if st.button("Start Homogeneous!"):
            width = dimension * 15
            height = dimension * 15
            screen = pygame.display.set_mode((width, height))
            CELL_SIZE = 15
            pygame.display.set_caption('Homogeneous Clustering')

            showBoard = Board(dimension, ant_amount, CORPSES_NUM, screen, CELL_SIZE, width, height)
            # dimension, ant_amount, corpse_amount, screen, cell_size, width, height):
            showBoard.cluster()

    with divider:
        st.markdown("<hr style='border: none; background-color: black; height: 465px; width: 1px; marginTop: -0.1px '>", unsafe_allow_html=True)

    with hete:
        hete_title = st.markdown("<h2 style='color: #6495ED;'>Heterogeneous Clustering</h2>", unsafe_allow_html=True)
        hete_ants = st.number_input("How Many Ants:", min_value=1, max_value=50, value=10, key="hete_ants")
        hete_foods = st.number_input("Quantity of food:", min_value=5, max_value=1000, value=75, key="hete_food")
        hete_board_dimension = st.number_input("Space size:", min_value=5, max_value=50, value=20, key="hete_board_dimension")

        if st.button("Start Heterogeneous!"):
            board_width = hete_board_dimension * 15
            board_height = hete_board_dimension * 15