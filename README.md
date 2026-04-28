# AI-Powered Conversational Learning Bot

AI-driven Telegram bot built with LLMs to guide users through structured, interactive learning flows.

## Overview

This project demonstrates how to design and implement a conversational system using large language models (LLMs), prompt engineering, and state-based workflows.

The bot guides users through content step-by-step:

* presents structured text fragments
* asks contextual questions
* adapts responses based on user input
* maintains conversation state across interactions

## Key Features

* LLM-powered conversational logic (OpenAI API)
* Prompt engineering for controlled AI behavior
* State management for multi-step user flows
* Support for both open-ended and multiple-choice questions
* Adaptive response handling (supportive, non-evaluative tone)
* Skip mechanism ("stop" command)
* Structured multi-stage workflow

## System Design

The project separates:

**1. Conversation Engine**

* dialogue flow
* state tracking
* user input handling
* transition logic

**2. Content Layer**

* structured JSON with text fragments
* questions and answer options
* learning goals per block

This allows reusing the same logic for different content domains.

## Tech Stack

* Python
* Telegram Bot API
* OpenAI API (LLM)
* JSON-based content structure

## Example Flow

1. User reads a short text fragment
2. Bot asks a question
3. User responds (free text or choice)
4. AI processes the response
5. Bot adapts and continues the flow

## Purpose

This project demonstrates practical skills in:

* building LLM-powered applications
* designing conversational UX
* implementing workflow-based systems
* integrating AI into real user interactions

## Notes

The current implementation uses literary content as a test domain, but the architecture is designed to be reusable for other structured learning scenarios.

## Status

MVP complete. Ready for testing, iteration, and extension.
